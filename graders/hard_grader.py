def grade_hard(state):
    try:
        err = state[6]
        queue = state[9]
    except Exception:
        err = state.get("error_rate", 1.0)
        queue = state.get("request_queue", 1000)

    score = 0
    if err < 0.1:
        score += 0.5
    if queue < 500:
        score += 0.5

    return score
