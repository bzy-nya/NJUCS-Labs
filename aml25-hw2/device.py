import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

from contextlib import nullcontext
# autocast = (
#     torch.autocast(device_type="mps", dtype=torch.float16)
#     if torch.backends.mps.is_available()
#     else nullcontext()
# )
autocast = nullcontext() # Use full precision
