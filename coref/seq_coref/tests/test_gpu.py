from mica_text_coref.coref.seq_coref import util
import torch
import getpass

user = getpass.getuser()
devices = list(range(torch.cuda.device_count()))
if devices:
    util.print_gpu_usage(user, devices)