def get_easy_task():
    return {
        "name": "easy_traffic_spike_recovery",
        "description": "Recover service quality during a traffic_spike using throttling and load rebalance.",
        "initial_state": {
            "scenario": "traffic_spike",
            "traffic_load": 0.95,
            "frontend_status": 2,
            "backend_status": 1,
            "db_status": 2,
            "request_queue": 650.0,
        },
        "max_steps": 24,
    }
