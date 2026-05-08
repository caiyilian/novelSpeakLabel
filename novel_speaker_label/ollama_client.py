from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:32b"
    timeout: int = 1800
    temperature: float = 0.0
    num_predict: int = 4096
    think: bool = False
    stream: bool = True


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": self.config.stream,
            "think": self.config.think,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.config.host.rstrip('/')}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._socket_timeout()
            ) as response:
                if self.config.stream:
                    return collect_streaming_response(response)
                body = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                "Ollama request timed out for "
                f"model {self.config.model} after {self.config.timeout} seconds. "
                "Use --timeout 0 to disable the socket timeout for long local runs."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for model {self.config.model}") from exc

        result = json.loads(body)
        return result.get("response", "")

    def _socket_timeout(self) -> int | None:
        if self.config.timeout <= 0:
            return None
        return self.config.timeout


def collect_streaming_response(lines: Iterable[bytes]) -> str:
    chunks: list[str] = []
    for raw_line in lines:
        if not raw_line.strip():
            continue
        message = json.loads(raw_line.decode("utf-8"))
        if "error" in message:
            raise RuntimeError(f"Ollama returned an error: {message['error']}")
        chunks.append(message.get("response", ""))
        if message.get("done"):
            break
    return "".join(chunks)
