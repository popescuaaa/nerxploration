import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    # device memory
    print(torch.cuda.memory_allocated())
    