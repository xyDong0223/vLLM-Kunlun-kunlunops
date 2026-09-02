# Installation

This document describes how to build and install vLLM-Kunlun manually in the
Baidu intranet environment.

## Requirements

| Component | Required version or value |
| --- | --- |
| Operating system | Ubuntu 20.04 |
| Python | 3.10 or later |
| Base image | `iregistry.baidu-int.com/hac_test/aiak-inference-llm:vLLM-Kunlun-Base` |
| PyTorch | `2.9.0` |
| vLLM | `0.25.0` |

## Set up the container environment

Use the following base image from the Baidu internal registry. The registry and
image are reachable only from the Baidu intranet.

### Container startup script

Save the following script as `start_docker.sh`, then run
`bash start_docker.sh <container_name>`.

```bash
#!/bin/bash

XPU_NUM=8
DOCKER_DEVICE_CONFIG=""
if [ "$XPU_NUM" -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM - 1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi

export build_image="iregistry.baidu-int.com/hac_test/aiak-inference-llm:vLLM-Kunlun-Base"

docker run -itd ${DOCKER_DEVICE_CONFIG} \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    -v /home/users/vllm-kunlun:/home/vllm-kunlun \
    --name "$1" \
    -w /workspace \
    "$build_image" /bin/bash
```

Enter the running container before continuing:

```bash
docker exec -it <container_name> /bin/bash
```

## Build and install vLLM-Kunlun

Run the following commands in the container. The sequence installs the required
PyTorch and vLLM versions, obtains the vLLM-Kunlun source, and builds the
package from source.

### Install PyTorch and vLLM

```bash
uv pip install torch==2.9.0 torchvision torchaudio
uv pip install vllm==0.25.0 --no-build-isolation --no-deps
```

### Clone, build, and install vLLM-Kunlun

```bash
git clone https://github.com/baidu/vLLM-Kunlun.git
cd vLLM-Kunlun
uv pip install -r requirements.txt
python setup.py build
python setup.py install
```

## Install XPytorch for Torch 2.9.0

After building and installing vLLM-Kunlun, return to the parent directory and
install the KL3-customized XPytorch package for Python 3.10 and Torch 2.9.0.
The `sed` command changes the installer to use `uv pip` and the active virtual
environment variable. Run this command only on a freshly unpacked installer.

```bash
cd ..
wget -O xpytorch-cp310-torch290-ubuntu2004-x64.run \
    https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/xpytorch-cp310-torch290-ubuntu2004-x64.run
bash xpytorch-cp310-torch290-ubuntu2004-x64.run --noexec --target xpytorch_unpack && cd xpytorch_unpack/ && \
    sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh && bash setup.sh
```

> **Compatibility note:** `vllm_kunlun/patches/patch_torch251.py` is specific to
> Torch 2.5.1 and must not be applied in this Torch 2.9.0 environment.

## Optional vLLM-Kunlun patches

If the deployment requires the vLLM-Kunlun patch files, apply them after the
installation steps above. These commands assume Python 3.10 and an active
virtual environment.

### Replace `eval_frame.py`

```bash
cp vllm_kunlun/patches/eval_frame.py \
    "${CONDA_PREFIX:-$VIRTUAL_ENV}"/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
```

### Replace quantization `__init__.py`

```bash
cp vllm_kunlun/quantization/__init__.py \
    "${CONDA_PREFIX:-$VIRTUAL_ENV}"/lib/python3.10/site-packages/vllm/model_executor/layers/quantization/__init__.py
```

## Install Kunlun-related packages

```bash
# Install kunlun_ops
uv pip install "https://baidu-kunlun-customer.su.bcebos.com/aiak/mimo/20260227/kunlun_ops-0.1.58+ee39020a-cp310-cp310-linux_x86_64.whl"

# Install xspeedgate_ops
uv pip install "https://vllm-ai-models.bj.bcebos.com/aiak_share/20260403/xspeedgate_ops-1.1.0+53992ca-cp310-cp310-linux_x86_64.whl"

# Install cocopod
uv pip install "https://vllm-ai-models.bj.bcebos.com/aiak_share/20260403/cocopod-1.1.0-cp310-cp310-linux_x86_64.whl"
```

## Quick start

### Set up the environment

```bash
chmod +x /workspace/vLLM-Kunlun/setup_env.sh
source /workspace/vLLM-Kunlun/setup_env.sh
```

### Run the server

```bash
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
      --served-model-name Qwen3-VL-30B-A3B-Instruct
```
