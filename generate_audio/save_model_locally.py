import torch
from stable_audio_tools import get_pretrained_model

# Initialize and move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)

# Path to save the combined model and configuration
save_path = "../stable_audio_model.pth"

# Save model and config as a tuple
torch.save((model.state_dict(), model_config), save_path)

print(f"Model and configuration saved as a tuple in {save_path}")
