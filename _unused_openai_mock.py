from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _join_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{suffix}"
    return f"{base}/v1{suffix}"


@dataclass
class _ChatCompletions:
    base_url: str
    api_key: str

    def create(self, *, model: str, messages: List[Dict[str, Any]], max_tokens: int = 5) -> Dict[str, Any]:
        url = _join_url(self.base_url, "/chat/completions")
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw}


@dataclass
class _Chat:
    completions: _ChatCompletions


class OpenAI:
    def __init__(self, *, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _Chat(completions=_ChatCompletions(base_url=base_url, api_key=api_key))
