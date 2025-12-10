# Installation

This document describes how to install vllm-kunlun manually.

## Requirements

- **OS**: Ubuntu 22.04 
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.5.1
  - vLLM (same version as vllm-kunlun)

## Setup environment using container
We provide a clean, minimal base image for your use`iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.32`.You can pull it using the `docker pull` command.
### Container startup script

:::::{tab-set}
:sync-group: install

::::{tab-item} start_docker.sh
:selected:
:sync: pip
```{code-block} bash
   :substitutions:
#!/bin/bash
XPU_NUM=8
DOCKER_DEVICE_CONFIG=""
if [ $XPU_NUM -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM-1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi
export build_image="iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.32"
docker run -itd ${DOCKER_DEVICE_CONFIG} \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    --cap-add=SYS_PTRACE \
    -v /home/users/vllm-kunlun:/home/vllm-kunlun \
    -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
    --name "$1" \
    -w /workspace \
    "$build_image" /bin/bash
```
::::
:::::
## Install vLLM-kunlun
### Install vLLM 0.11.0
```
conda activate python310_torch25_cuda

pip install vllm==0.11.0 
```
### Build and Install
Navigate to the vllm-kunlun directory and build the package:
```
git clone xxxx # TODO: replace with Github Url to install vllm-kunlun

cd vllm-kunlun

pip install -r requirements.txt

python setup.py build

python setup.py install

```
### Replace eval_frame.py
Copy the eval_frame.py patch:
```
cp vllm_kunlun/patches/eval_frame.py /root/miniconda/envs/python310_torch25_cuda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
```
## Install the KL3-customized build of PyTorch
```
wget https://klx-sdk-release-public.su.bcebos.com/xpytorch/release/3.3.2.7/xpytorch-cp310-torch251-ubuntu2004-x64.run && bash xpytorch-cp310-torch251-ubuntu2004-x64.run
```

## Install custom ops
```
pip uninstall xtorch_ops -y && pip install \
"https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/xtorch_ops-0.1.2028%2B1baf1b15-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-10-31T10%3A38%3A24Z%2F-1%2Fhost%2Faa1969b70a4a97c407d69614a5d5a3e26ea07286d13f0a2ab8daccc288152903"
```

## Install the KLX3 custom Triton build
```
pip install \
"https://cce-ai-models.bj.bcebos.com/v1/vllm-kunlun-0.11.0/triton-3.0.0%2Bb2cde523-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKxPW2jzoJUuFZmI19s3yry%2F2025-11-05T02%3A47%3A29Z%2F-1%2Fhost%2Fd8c95dbd06187a3140ca3e681e00c6941c30e14bb1d4112a0c8bc3c93e5c9c3f"
```
## Install the AIAK custom ops library
```
pip install \
"https://cce-ai-models.bj.bcebos.com/v1/chenyili/xspeedgate_ops-0.0.0-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKxPW2jzoJUuFZmI19s3yry%2F2025-11-18T01%3A56%3A21Z%2F-1%2Fhost%2F28b57cbc5dc62ac1bf946e74146b3ea4952d2ffff448617f0303980dcaf6cb49"
```
## Quick Start

### Set up the environment

```
chmod +x /workspace/baidu/hac-aiacc/vllm-kunlun/setup_env.sh && source /workspace/baidu/hac-aiacc/vllm-kunlun/setup_env.sh
```

### Run the server
:::::{tab-set}
:sync-group: install

::::{tab-item} start_service.sh
:selected:
:sync: pip
```{code-block} bash
   :substitutions:
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8356 \
      --model models/Qwen3-VL-30B-A3B-Instruct \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-VL-30B-A3B-Instruct \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention", 
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2", 
                                                "vllm.mamba_mixer", 
                                                "vllm.short_conv", 
                                                "vllm.linear_attention", 
                                                "vllm.plamo2_mamba_mixer", 
                                                "vllm.gdn_attention", 
                                                "vllm.sparse_attn_indexer"]}' \  

```
::::
:::::