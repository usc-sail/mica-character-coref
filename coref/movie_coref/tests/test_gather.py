import accelerate
from accelerate import logging
import logging as _logging
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def test_gather():
    accelerator = accelerate.Accelerator()
    logger = logging.get_logger("")
    if accelerator.is_local_main_process:
        log_file = "/home/sbaruah_usc_edu/mica_text_coref/data/movie_coref/results/coreference/logs/test.log"
        file_handler = _logging.FileHandler(log_file, mode="w")
        logger.logger.addHandler(file_handler)
    n_processes = accelerator.num_processes
    batch_size = 5
    n_iters = 3
    x = torch.randn(batch_size * n_processes * n_iters, 10)
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = nn.Linear(10, 1)
    dataloader, model = accelerator.prepare(dataloader, model)
    for i, batch in enumerate(dataloader):
        inp = batch[0]
        logger.info(f"Batch {i + 1}: input (before forward) = {inp.shape} ({inp.device})", main_process_only=False)
        out = model(inp)
        logger.info(f"Batch {i + 1}: output (after forward) = {out.shape} ({out.device})", main_process_only=False)
        out = accelerator.gather_for_metrics(out)
        logger.info(f"Batch {i + 1}: output (after gatherm) = {out.shape} ({out.device})", main_process_only=False)

if __name__=="__main__":
    test_gather()