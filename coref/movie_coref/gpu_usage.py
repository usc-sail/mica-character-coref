"""Calculate GPU memory allocated by function"""
import torch

class GPUTracker:
    """Track the maximum memory allocated"""
    def __init__(self, device: torch.device | int = None) -> None:
        self._device = device
        self._max_mem = 0

    def __enter__(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self._device)
    
    def __exit__(self, type, value, traceback):
        self._max_mem = torch.cuda.max_memory_allocated(self._device)
    
    @property
    def max_memory(self) -> float:
        return self._max_mem/10**9