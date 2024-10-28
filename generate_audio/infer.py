import os
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to calculate model size in MB
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

# Set path to save/load the model and configuration together
model_path = "../stable_audio_model.pth"

# Check if model is already saved locally
if os.path.exists(model_path):
    # Load the model and configuration together from a single file
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint["model"]
    model_config = checkpoint["model_config"]
else:
    # Download model and configuration
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    
    # Save model and configuration together in a dictionary
    torch.save({"model": model, "model_config": model_config}, model_path)

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "128 BPM tech house drum loop",
    "seconds_start": 0, 
    "seconds_total": 30
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)
