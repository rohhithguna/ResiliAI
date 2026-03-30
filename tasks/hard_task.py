def get_hard_task():
    return {
        "name": "multi_failure_recovery",
        "description": "Handle multiple failures with traffic and queue buildup",
        "initial_state": {
            "db_status": 0,
            "backend_status": 1,
            "traffic_load": 0.95,
            "request_queue": 800
        },
        "goal": "full recovery with stable metrics",
        "max_steps": 40
    }
