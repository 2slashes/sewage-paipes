from pipes_nn_classes import PipesPredictor
import torch
from torch import nn
import os

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

n = 4
model = PipesPredictor(n**2 * 4, 64, n**2).to(device)

model_file_name = "model.pth"
curr_dir = os.path.dirname(__file__)
model_file_path = os.path.join(curr_dir, model_file_name)
model.load_state_dict(torch.load(model_file_path, weights_only=True))
model.eval()


state = "0100010101110001010001011011001001000011100110110100110101011001"
state_int_list = [int(x) for x in state]
state_tensor = torch.tensor(state_int_list).to(device).float()
print(nn.functional.softmax(state_tensor, dim=1))
print(torch.argmax(model(state_tensor)))
