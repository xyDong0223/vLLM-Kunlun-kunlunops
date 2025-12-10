# Single XPU (Qwen3-8B)

## Run vllm-kunlun on Single XPU

Setup environment using container:

```bash
# !/bin/bash
# rundocker.sh
XPU_NUM=8
DOCKER_DEVICE_CONFIG=""
if [ $XPU_NUM -gt 0 ]; then
    for idx in $(seq 0 $((XPU_NUM-1))); do
        DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpu${idx}:/dev/xpu${idx}"
    done
    DOCKER_DEVICE_CONFIG="${DOCKER_DEVICE_CONFIG} --device=/dev/xpuctrl:/dev/xpuctrl"
fi

export build_image="xxxxxxxxxxxxxxxxx" 

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

### Offline Inference on Single XPU

Start the server in a container:

```{code-block} bash
from vllm import LLM, SamplingParams

def main():

    model_path = "/models/Qwen3-8B"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
        "dtype": "float16",
        "enable_chunked_prefill": False,
        "distributed_executor_backend": "mp",
    }

    llm = LLM(**llm_params)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "说个笑话"
                }
            ]
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_token_ids=[181896]
    )

    outputs = llm.chat(messages, sampling_params=sampling_params)

    response = outputs[0].outputs[0].text
    print("=" * 50)
    print("输入内容:", messages)
    print("模型回复:\n", response)
    print("=" * 50)

if __name__ == "__main__":
    main()

```

:::::

If you run this script successfully, you can see the info shown below:

```bash
==================================================
输入内容: [{'role': 'user', 'content': [{'type': 'text', 'text': '说个笑话'}]}]
模型回复:
 <think>
好的，用户让我讲个笑话。首先，我需要考虑用户的需求。他们可能只是想轻松一下，或者需要一些娱乐。接下来，我要选择一个适合的笑话，不要太复杂，容易理解，同时也要有趣味性。

用户可能希望笑话是中文的，所以我要确保笑话符合中文的语言习惯和文化背景。我需要避免涉及敏感话题，比如政治、宗教或者可能引起误解的内容。然后，我得考虑笑话的结构，通常是一个设置和一个出人意料的结尾，这样能带来笑点。

例如，可以讲一个关于日常生活的小幽默，比如动物或者常见的场景。比如，一只乌龟和兔子赛跑的故事，但加入一些反转。不过要确保笑话的长度适中，不要太长，以免用户失去兴趣。另外，要注意用词口语化，避免生硬或复杂的句子结构。

可能还要检查一下这个笑话是否常见，避免重复。如果用户之前听过类似的，可能需要
==================================================
```

### Online Serving on Single XPU

Start the vLLM server on a single XPU:

```{code-block} bash
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 9000 \
      --model /models/Qwen3-8B\
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 32768 \
      --max-seq-len-to-capture 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-8B \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention_with_output_kunlun",
            "vllm.unified_attention", "vllm.unified_attention_with_output",
            "vllm.mamba_mixer2"]}' \
```
If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=118459) INFO:     Started server process [118459]
(APIServer pid=118459) INFO:     Waiting for application startup.
(APIServer pid=118459) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:9000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-8B",
        "prompt": "What is your name?",
        "max_tokens": 100,
        "temperature": 0
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-80ee8b893dc64053947b0bea86352faa","object":"text_completion","created":1763015742,"model":"Qwen3-8B","choices":[{"index":0,"text":" is the S, and ,","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
(APIServer pid=54567) INFO:     127.0.0.1:60338 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=54567) INFO 11-13 14:35:48 [loggers.py:123] Engine 000: Avg prompt throughput: 0.5 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```