"""Model factory.

Any OpenAI-compatible endpoint works, which includes the usual ways to serve an open-source model:
Ollama (http://localhost:11434/v1), vLLM, llama.cpp server, LM Studio. The executor needs native
tool calling, so choose a model/server pair that supports it (for example Qwen2.5-Coder or Qwen3
on Ollama or vLLM with tool parsing enabled).

Environment variables (all optional except that a model name is needed):
  LLM_MODEL     model name as the server knows it
  LLM_BASE_URL  default http://localhost:11434/v1
  LLM_API_KEY   default "ollama" (local servers ignore it)
"""
from __future__ import annotations

import os


def build_llm(model: str | None = None, base_url: str | None = None,
              api_key: str | None = None, temperature: float = 0.2,
              timeout_s: float = 600.0):
    from langchain_openai import ChatOpenAI

    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("Set LLM_MODEL (or pass --model), e.g. LLM_MODEL=qwen2.5-coder:14b")
    return ChatOpenAI(
        model=model,
        base_url=base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=api_key or os.environ.get("LLM_API_KEY", "ollama"),
        temperature=temperature,
        timeout=timeout_s,
        max_retries=2,
    )
