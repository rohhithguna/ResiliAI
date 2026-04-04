def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _extract_values(result):
    if isinstance(result, dict):
        return {
            "initial_error": _safe_float(result.get("initial_error", 1.0), 1.0),
            "final_error": _safe_float(result.get("final_error", result.get("error_rate", 1.0)), 1.0),
            "request_queue": _safe_float(result.get("request_queue", 1000.0), 1000.0),
            "steps": int(_safe_float(result.get("steps", 999), 999)),
            "max_steps": int(_safe_float(result.get("max_steps", 999), 999)),
            "recovered": bool(result.get("recovered", False)),
            "base_score": _safe_float(result.get("score", result.get("final_score", 0.0)), 0.0),
        }
    if isinstance(result, (list, tuple)):
        err = _safe_float(result[6], 1.0) if len(result) > 6 else 1.0
        queue = _safe_float(result[9], 1000.0) if len(result) > 9 else 1000.0
        return {
            "initial_error": 1.0,
            "final_error": err,
            "request_queue": queue,
            "steps": 999,
            "max_steps": 999,
            "recovered": False,
            "base_score": 0.0,
        }
    return {
        "initial_error": 1.0,
        "final_error": 1.0,
        "request_queue": 1000.0,
        "steps": 999,
        "max_steps": 999,
        "recovered": False,
        "base_score": 0.0,
    }


def grade_hard(result):
    v = _extract_values(result)

    error_reduction = _clamp01(v["initial_error"] - v["final_error"])
    recovery_score = 1.0 if v["recovered"] else 0.0
    efficiency = _clamp01(1.0 - (max(0, v["steps"]) / max(1, v["max_steps"])))
    queue_score = _clamp01(1.0 - (v["request_queue"] / 1000.0))

    score = (
        0.40 * error_reduction
        + 0.30 * recovery_score
        + 0.15 * efficiency
        + 0.10 * queue_score
        + 0.05 * _clamp01(v["base_score"])
    )
    return float(_clamp01(score))
