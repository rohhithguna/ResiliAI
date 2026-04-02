def grade_medium(state):
    try:
        err = state[6]
        traffic = state[7]
    except Exception:
        err = state.get("error_rate", 1.0)
        traffic = state.get("traffic_load", 1.0)

    score = 0
    if err < 0.1:
        score += 0.5
    if traffic < 0.7:
        score += 0.5

    return score
