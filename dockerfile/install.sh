#!/bin/bash

set -exuo pipefail

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch25_cuda
echo 'conda activate python310_torch25_cuda' >> ~/.bashrc
echo 'source /workspace/vllm-kunlun/setup_env.sh' >> ~/.bashrc

#安装社区vllm
cd /workspace/vllm-kunlun
pip uninstall vllm -y
pip uninstall vllm-kunlun -y
pip install vllm==0.11.0 --no-build-isolation --no-deps --index-url https://pip.baidu-int.com/simple/

#
pip install -r /workspace/vllm-kunlun/requirements.txt

#安装vllm-kunlun
python setup.py build
python setup.py install
cp vllm_kunlun/patches/eval_frame.py /root/miniconda/envs/python310_torch25_cuda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py

#安装Kl3自定义torch 01130
wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/1130/xpytorch-cp310-torch251-ubuntu2004-x64.run?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-12-02T05%3A01%3A27Z%2F-1%2Fhost%2Ff3cf499234f82303891aed2bcb0628918e379a21e841a3fac6bd94afef491ff7
bash xpytorch-cp310-torch251-ubuntu2004-x64.run
rm xpytorch-cp310-torch251-ubuntu2004-x64.run
#安装Klx3自定义算子库 01130
pip uninstall xtorch_ops -y
pip install "https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/1130/xtorch_ops-0.1.2209%2B6752ad20-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-12-05T06%3A18%3A00Z%2F-1%2Fhost%2F14936c2b7e7c557c1400e4c467c79f7a9217374a7aa4a046711ac4d948f460cd"
#安装klx3自定义triton
pip install "https://cce-ai-models.bj.bcebos.com/v1/vllm-kunlun-0.11.0/triton-3.0.0%2Bb2cde523-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKxPW2jzoJUuFZmI19s3yry%2F2025-11-05T02%3A47%3A29Z%2F-1%2Fhost%2Fd8c95dbd06187a3140ca3e681e00c6941c30e14bb1d4112a0c8bc3c93e5c9c3f"
#安装AIAK自定义算子库
pip install "https://cce-ai-models.bj.bcebos.com/v1/chenyili/xspeedgate_ops-0.0.0-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKxPW2jzoJUuFZmI19s3yry%2F2025-12-05T06%3A37%3A39Z%2F-1%2Fhost%2F1002777dadd2afe4c1f047cbf0d94244d5b1f03295cd8f7a2802b92a13cd5035"