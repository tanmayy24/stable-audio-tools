import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import time
import os
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight

# Function to calculate model size in MB
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and configuration
model_path = "../stable_audio_model.pth"
checkpoint = torch.load(model_path, map_location=device)
model = checkpoint["model"]
model_config = checkpoint["model_config"]
print_size_of_model(model)

# Apply dynamic quantization
quantize_(model, int8_dynamic_activation_int8_weight())

print_size_of_model(model)

# Move model to the GPU if not already there
model = model.to(device)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

# Set up text and timing conditioning
conditioning = [{
    "prompt": "A dog is barking in a busy street.",
    "seconds_start": 0,
    "seconds_total": 10
}]

# Start timing
start_time = time.time()

# Generate stereo audio on GPU
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device  # Ensure GPU is used
)

# End timing
end_time = time.time()
generation_time = end_time - start_time
print(f"Time to generate audio: {generation_time:.2f} seconds")

# Rearrange audio batch to a single sequence on GPU
output = rearrange(output, "b d n -> d (b n)").to(device)

# Peak normalize, clip, convert to int16, and save to file on GPU
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("stable_audio_dog.wav", output, sample_rate)