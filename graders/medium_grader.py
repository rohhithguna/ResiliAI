def grade_medium(state):
    latency_ok = (
        state["frontend_latency"] < 300 and
        state["backend_latency"] < 300 and
        state["db_latency"] < 300
    )

    error_ok = state["error_rate"] < 0.05

    score = 0.0

    if latency_ok:
        score += 0.5
    if error_ok:
        score += 0.5

    return score
