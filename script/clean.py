import torch

def clean_gpu_memory():
    torch.cuda.empty_cache()