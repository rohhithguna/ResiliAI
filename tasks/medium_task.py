def get_medium_task():
    return {
        "name": "medium_db_failure_recovery",
        "description": "Handle db_failure with cascading backend latency and restore stable service.",
        "initial_state": {
            "scenario": "db_failure",
            "frontend_status": 2,
            "backend_status": 1,
            "db_status": 0,
            "backend_latency": 1200.0,
            "db_latency": 1900.0,
            "traffic_load": 0.8,
            "request_queue": 720.0,
        },
        "max_steps": 30,
    }
