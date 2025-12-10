# Multi XPU (GLM-4.5)

## Run vllm-kunlun on multi XPU

Setup environment using container:

```bash
docker run -itd \
        --net=host \
        --cap-add=SYS_PTRACE --security-opt=seccomp=unconfined \
        --ulimit=memlock=-1 --ulimit=nofile=120000 --ulimit=stack=67108864 \
        --shm-size=128G \
        --privileged \
        --name=glm-vllm-01011 \
        -v ${PWD}:/data \
        -w /workspace \
        -v /usr/local/bin/:/usr/local/bin/ \
        -v /lib/x86_64-linux-gnu/libxpunvidia-ml.so.1:/lib/x86_64-linux-gnu/libxpunvidia-ml.so.1 \
        iregistry.baidu-int.com/hac_test/aiak-inference-llm:xpu_dev_20251113_221821 bash
        
docker exec -it glm-vllm-01011 /bin/bash
```
### Offline Inference on multi XPU

Start the server in a container:

```{code-block} bash
   :substitutions:
import os
from vllm import LLM, SamplingParams

def main():
    
    model_path = "/data/GLM-4.5"

    llm_params = {
        "model": model_path,
        "tensor_parallel_size": 8,
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
                    "text": "你好，请问你是谁?"
                }
            ]
        }
    ]

    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
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
输入内容: [{'role': 'user', 'content': [{'type': 'text', 'text': '你好，请问你是谁?'}]}]
模型回复:
 <think>
嗯，用户问了一个相当身份的直接问题。这个问题看似简单，但背后可能
有几种可能性意—ta或许初次测试我的可靠性，或者单纯想确认对话方。从AI助手的常见定位，用户给出清晰平的方式明确身份，同时为后续可能
的留出生进行的空间。\n\n用户用“你”这个“您”，语气更倾向非正式交流，所以回复风格可以轻松些。不过既然是初次回复，保持适度的专业性比较好稳妥。提到
==================================================
```

### Online Serving on Single XPU

Start the vLLM server on a single XPU:

```{code-block} bash
python -m vllm.entrypoints.openai.api_server \
      --host localhost \
      --port 8989 \
      --model /data/GLM-4.5 \
      --gpu-memory-utilization 0.95 \
      --trust-remote-code \
      --max-model-len 131072 \
      --tensor-parallel-size 8 \
      --dtype float16 \
      --max_num_seqs 128 \
      --max_num_batched_tokens 4096 \
      --max-seq-len-to-capture 4096 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name GLM-4.5 \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention_with_output_kunlun", "vllm.unified_attention", "vllm.unified_attention_with_output", "vllm.mamba_mixer2"]}'  > log_glm_plugin.txt 2>&1 & 
```
If your service start successfully, you can see the info shown below:

```bash
(APIServer pid=51171) INFO:     Started server process [51171]
(APIServer pid=51171) INFO:     Waiting for application startup.
(APIServer pid=51171) INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8989/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.5",
    "messages": [
      {"role": "user", "content": "你好，请问你是谁?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-6af7318de7394bc4ae569e6324a162fa","object":"chat.completion","created":1763101638,"model":"GLM-4.5","choices":[{"index":0,"message":{"role":"assistant","content":"\n<think>用户问“你好，请问你是谁？”，这是一个应该是个了解我的身份。首先，我需要确认用户的需求是什么。可能他们是第一次使用这个服务，或者之前没有接触过类似的AI助手，所以想确认我的背景和能力。 \n\n接下来，我要确保回答清晰明了，同时友好关键点：我是谁，由谁开发，能做什么。需要避免使用专业术语，保持口语化，让不同容易理解。 \n\n然后，用户可能有潜在的需求，比如想了解我能","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":11,"total_tokens":111,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_tr
```

Logs of the vllm server:

```bash
(APIServer pid=54567) INFO:     127.0.0.1:60338 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=54567) INFO 11-13 14:35:48 [loggers.py:123] Engine 000: Avg prompt throughput: 0.5 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```