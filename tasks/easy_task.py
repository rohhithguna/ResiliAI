def get_easy_task():
    return {
        "name": "recover_single_service",
        "description": "Recover system from single service failure",
        "initial_state": {
            "db_status": 0
        },
        "goal": "all services healthy",
        "max_steps": 20
    }
