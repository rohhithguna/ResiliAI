def grade_easy(state):
    try:
        err = state[6]
    except Exception:
        err = state.get("error_rate", 1.0)

    return 1.0 if err < 0.05 else 0.0
