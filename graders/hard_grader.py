def grade_hard(state):
    health = (
        state["frontend_status"] == 2 and
        state["backend_status"] == 2 and
        state["db_status"] == 2
    )

    latency = (
        state["frontend_latency"] < 300 and
        state["backend_latency"] < 300 and
        state["db_latency"] < 300
    )

    error = state["error_rate"] < 0.03
    queue = state["request_queue"] < 100

    score = 0.0

    if health:
        score += 0.4
    if latency:
        score += 0.2
    if error:
        score += 0.2
    if queue:
        score += 0.2

    return score
