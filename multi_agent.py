def _safe_get(state, idx, key, default):
    if isinstance(state, dict):
        return state.get(key, default)
    if isinstance(state, (list, tuple)) and idx < len(state):
        return state[idx]
    return default


def _state_to_obs(state):
    return {
        "frontend_status": int(_safe_get(state, 0, "frontend_status", 0)),
        "backend_status": int(_safe_get(state, 1, "backend_status", 0)),
        "db_status": int(_safe_get(state, 2, "db_status", 0)),
        "frontend_latency": float(_safe_get(state, 3, "frontend_latency", 0.0)),
        "backend_latency": float(_safe_get(state, 4, "backend_latency", 0.0)),
        "db_latency": float(_safe_get(state, 5, "db_latency", 0.0)),
        "error_rate": float(_safe_get(state, 6, "error_rate", 1.0)),
        "traffic_load": float(_safe_get(state, 7, "traffic_load", 0.0)),
        "traffic_balance": float(_safe_get(state, 8, "traffic_balance", 0.0)),
        "request_queue": float(_safe_get(state, 9, "request_queue", 0.0)),
    }


class FrontendAgent:
    def suggest(self, obs):
        if obs["frontend_status"] == 0 or obs["frontend_latency"] > 800:
            return 1
        return None


class BackendAgent:
    def suggest(self, obs):
        if obs["backend_status"] == 0 or obs["backend_latency"] > 800:
            return 2
        return None


class DatabaseAgent:
    def suggest(self, obs):
        if obs["db_status"] == 0 or obs["db_latency"] > 800:
            return 3
        return None


class TrafficAgent:
    def suggest(self, obs):
        if obs["traffic_load"] > 0.85 or obs["error_rate"] > 0.3:
            return 4
        return None


class QueueAgent:
    def suggest(self, obs):
        if obs["request_queue"] > 400:
            return 5
        return None


class CoordinatorAgent:
    def __init__(self):
        self.frontend = FrontendAgent()
        self.backend = BackendAgent()
        self.database = DatabaseAgent()
        self.traffic = TrafficAgent()
        self.queue = QueueAgent()

    def select_action(self, state):
        obs = _state_to_obs(state)

        for agent in [
            self.database,
            self.backend,
            self.frontend,
            self.traffic,
            self.queue,
        ]:
            action = agent.suggest(obs)
            if action is not None:
                return action

        return 0
