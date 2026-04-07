def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _extract_error(result):
    if isinstance(result, dict):
        if "final_error" in result:
            return _safe_float(result.get("final_error"), 1.0)
        if "error_rate" in result:
            return _safe_float(result.get("error_rate"), 1.0)
    if isinstance(result, (list, tuple)) and len(result) > 6:
        return _safe_float(result[6], 1.0)
    return 1.0


def grade_easy(result):
    if not isinstance(result, dict):
        result = {"error_rate": _extract_error(result)}

    initial_error = _safe_float(result.get("initial_error", 1.0), 1.0)
    final_error = _extract_error(result)
    steps = int(_safe_float(result.get("steps", 999), 999))
    max_steps = int(_safe_float(result.get("max_steps", max(steps, 1)), max(steps, 1)))
    recovered = bool(result.get("recovered", final_error <= 0.1))
    base_score = _safe_float(result.get("score", result.get("final_score", 0.0)), 0.0)

    error_reduction = _clamp01(initial_error - final_error)
    recovery_score = 1.0 if recovered else 0.0
    efficiency = _clamp01(1.0 - (max(0, steps) / max(1, max_steps)))

    score = 0.55 * error_reduction + 0.30 * recovery_score + 0.10 * efficiency + 0.05 * _clamp01(base_score)
    score = max(1e-6, min(score, 0.999999))
    return float(_clamp01(score))
