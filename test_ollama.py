import requests
import json


# Ollama 服务地址
OLLAMA_HOST = "http://127.0.0.1:11434"
# 多模型同时决策，越可靠的模型越靠前
total_modelname = ["qwen3:32b","qwen3-coder:30b","qwen2.5-coder:32b","qwen2.5-coder:14b","llama3:latest","glm-4.7-flash:latest"]
def test_ollama_generate(my_prompt, MODEL="qwen2.5-coder:32b"):
    """
    向 Ollama API 发送一个生成请求
    """
    # API 端点
    url = f"{OLLAMA_HOST}/api/generate"
    # 请求负载
    payload = {
        "model": MODEL,
        "prompt": my_prompt,
        "stream": False,  # 设置为 True 可接收流式响应，但是没啥太大的用处，所以设置成False加快速度
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 32700
        }
    }
    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    response.raise_for_status()  # 如果响应状态码不是 200，将抛出 HTTPError 异常

    # 解析响应
    result = response.json()
    return result.get('response', 'No response content')
