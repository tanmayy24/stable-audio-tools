import torch
import torchaudio
import pandas as pd
import os
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import math
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Function to initialize distributed process
def setup(rank, world_size):
    # Set environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'  # Use 'localhost' for local training
    os.environ['MASTER_PORT'] = '29500'      # Use any available port
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Function to cleanup distributed process
def cleanup():
    dist.destroy_process_group()

# Load the model and config separately
def load_model(model_path):
    model_tuple = torch.load(model_path, map_location="cuda:0")
    model, config = model_tuple[0], model_tuple[1]
    print_size_of_model(model)
    # Apply dynamic quantization
    quantize_(model, int8_dynamic_activation_int8_weight())
    print_size_of_model(model)
    return model, config

# Function to process a batch of data
def process_batch(batch_data, model, rank, sample_size):
    outputs = []
    for _, row in batch_data.iterrows():
        audiocap_id = row['audiocap_id']
        caption = row['caption']
        start_time = row['start_time']

        # Set up text and timing conditioning
        conditioning = [{
            "prompt": caption,
            "seconds_start": 0, 
            "seconds_total": 10  # Adjust as needed
        }]

        # Generate stereo audio
        with torch.no_grad():  
            output = generate_diffusion_cond(
                model.module, 
                steps=100,
                cfg_scale=7,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=f'cuda:{rank}'
            )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        outputs.append((audiocap_id, output))
    return outputs

# Function to run the distributed training process
def run(rank, world_size, csv_file_path, model_path, output_folder, batch_size):
    setup(rank, world_size)

    # Load model and move it to the current GPU
    model, config = load_model(model_path)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Load data
    data = pd.read_csv(csv_file_path)
    sample_rate = config["sample_rate"]
    sample_size = config["sample_size"]

    # Split data across GPUs by slicing
    num_samples_per_rank = len(data) // world_size
    if rank == world_size - 1:
        data_per_rank = data.iloc[rank * num_samples_per_rank:]
    else:
        data_per_rank = data.iloc[rank * num_samples_per_rank:(rank + 1) * num_samples_per_rank]

    num_batches = math.ceil(len(data_per_rank) / batch_size)

    os.makedirs(output_folder, exist_ok=True)

    # Process each batch
    for i in range(num_batches):
        batch_data = data_per_rank.iloc[i * batch_size:(i + 1) * batch_size]

        # Process batch
        outputs = process_batch(batch_data, model, rank, sample_size)

        # Save output
        for audiocap_id, output in outputs:
            output_file = os.path.join(output_folder, f"{audiocap_id}.wav")
            torchaudio.save(output_file, output, sample_rate)
            if rank == 0:
                print(f"Audio saved as {output_file}")

    cleanup()

    # After all processes finish, check for any missing files
    if rank == 0:
        generated_files = set([f.replace('.wav', '') for f in os.listdir(output_folder)])
        missing_audiocaps_ids = set(data['audiocap_id'].astype(str)) - generated_files

        if missing_audiocaps_ids:
            print(f"Missing audio for: {missing_audiocaps_ids}")
            # Optionally, reprocess missing samples (this would require a separate step or script)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # Arguments to pass to the distributed process
    csv_file_path = "../stable-audio-metrics/load/audiocaps_test_250.csv"
    model_path = "../stable_audio_model.pth"
    output_folder = "generated_output_250"
    batch_size = 64 

    # Use multiprocessing to spawn a process per GPU
    mp.spawn(
        run,
        args=(world_size, csv_file_path, model_path, output_folder, batch_size),
        nprocs=world_size,
        join=True
    )
