import torch

print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Should show 12.1
print(torch.cuda.get_device_name(0))  # Your GPU model
