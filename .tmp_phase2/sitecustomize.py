import atexit
import os
import sys
import threading

log_file = os.environ.get("TRACE_LOG_FILE")


def log(msg):
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    else:
        print(msg)


def trace(frame, event, arg):
    if event != "call":
        return trace
    mod = frame.f_globals.get("__name__", "")
    filename = frame.f_code.co_filename.rsplit("/", 1)[-1]
    name = frame.f_code.co_name
    if filename == "inference.py" and name == "run_task":
        log(f"RUN_TASK_ENTER {mod}")
    if filename == "inference.py" and name == "call_llm":
        log(f"CALL_LLM_ENTER {mod}")
    if mod == "__main__" and name == "run_all":
        log("ENTRY_FUNCTION run_all")
    if mod == "openai" and name == "__init__":
        self_obj = frame.f_locals.get("self")
        if self_obj is not None and self_obj.__class__.__name__ == "OpenAI":
            log("OPENAI_INIT")
    if mod == "openai" and name == "create":
        self_obj = frame.f_locals.get("self")
        if self_obj is not None and self_obj.__class__.__name__ == "_ChatCompletions":
            log("API_CREATE")
    return trace


sys.setprofile(trace)
threading.setprofile(trace)
