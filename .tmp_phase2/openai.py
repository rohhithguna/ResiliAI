import os
import json

_LOG = os.environ.get("TRACE_LOG_FILE")


def _log(line):
    if not _LOG:
        return
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class _ChatCompletions:
    def create(self, **kwargs):
        _log("API_CREATE")
        class Response:
            pass
        return Response()


class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()


class OpenAI:
    def __init__(self, *, base_url, api_key):
        _log(f"OPENAI_INIT base_url={base_url} api_key={api_key}")
        self.chat = _Chat()
