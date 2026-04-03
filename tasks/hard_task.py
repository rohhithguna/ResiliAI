def get_hard_task():
    return {
        "name": "hard_multi_failure_recovery",
        "description": "Recover from a multi_failure involving backend, database, and load-balancer instability.",
        "initial_state": {
            "scenario": "multi_failure",
            "frontend_status": 1,
            "backend_status": 0,
            "db_status": 1,
            "frontend_latency": 950.0,
            "backend_latency": 1350.0,
            "db_latency": 1250.0,
            "traffic_load": 0.86,
            "request_queue": 760.0,
            "traffic_balance": 0.9,
        },
        "max_steps": 40,
    }
