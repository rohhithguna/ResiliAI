def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _extract_values(result):
    if isinstance(result, dict):
        return (
            _safe_float(result.get("initial_error", 1.0), 1.0),
            _safe_float(result.get("final_error", result.get("error_rate", 1.0)), 1.0),
            _safe_float(result.get("traffic_load", 1.0), 1.0),
            int(_safe_float(result.get("steps", 999), 999)),
            int(_safe_float(result.get("max_steps", 999), 999)),
            bool(result.get("recovered", False)),
            _safe_float(result.get("score", result.get("final_score", 0.0)), 0.0),
        )
    if isinstance(result, (list, tuple)):
        err = _safe_float(result[6], 1.0) if len(result) > 6 else 1.0
        traffic = _safe_float(result[7], 1.0) if len(result) > 7 else 1.0
        return (1.0, err, traffic, 999, 999, False, 0.0)
    return (1.0, 1.0, 1.0, 999, 999, False, 0.0)


def grade_medium(result):
    initial_error, final_error, traffic_load, steps, max_steps, recovered, base_score = _extract_values(result)

    error_reduction = initial_error - final_error
    recovery_score = 1.0 if recovered else 0.0
    efficiency = 1.0 - (max(0, steps) / max(1, max_steps))
    traffic_score = 1.0 - traffic_load

    score = (
        0.45 * error_reduction
        + 0.25 * recovery_score
        + 0.15 * efficiency
        + 0.10 * traffic_score
        + 0.05 * base_score
    )
    return float(score)
