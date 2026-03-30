def get_medium_task():
    return {
        "name": "handle_latency_and_traffic",
        "description": "Reduce latency and error under high traffic",
        "initial_state": {
            "backend_latency": 1200,
            "traffic_load": 0.9
        },
        "goal": "latency < 300 and error_rate < 0.05",
        "max_steps": 30
    }
