import torch


def sync_and_print_mem_usage(label: str | None = None, device: int | torch.device | None = None) -> None:
    """Synchronize the selected GPU and print its currently allocated memory in MB."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    torch.cuda.synchronize(device)
    dev_index = device if device is not None else torch.cuda.current_device()
    allocated_mb = torch.cuda.memory_allocated(dev_index) / (1024 * 1024)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}GPU{dev_index} allocated: {allocated_mb:.2f} MB")
