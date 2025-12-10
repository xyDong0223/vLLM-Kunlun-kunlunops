"""worker"""
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from vllm.v1.worker.gpu_worker import Worker, _check_if_gpu_supports_dtype, init_worker_distributed_environment
from vllm.model_executor import set_random_seed
from .model_runner import KunlunModelRunner
from vllm.utils import MemorySnapshot
import torch
import os
import gc

class KunlunWorker(Worker):
    """Worker"""

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.init_snapshot = MemorySnapshot() 
            free_memory, total = torch.cuda.mem_get_info()
            self.init_gpu_memory = free_memory
            # 设置一个合理的初始值，比如总内存的 80%
            self.requested_memory = int(total * 0.2)  # 留出 20% 的余量
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, 
                                            self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        # Construct the model runner
        self.model_runner: KunlunModelRunner = KunlunModelRunner(
            self.vllm_config, self.device)
