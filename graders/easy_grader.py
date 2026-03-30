def grade_easy(state):
    healthy = (
        state["frontend_status"] == 2 and
        state["backend_status"] == 2 and
        state["db_status"] == 2
    )
    return 1.0 if healthy else 0.0
