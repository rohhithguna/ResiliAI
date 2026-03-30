import torch


class FrontendAgent:
    def suggest_action(self, state):
        frontend_status = int(state[0])
        frontend_latency = float(state[3])

        if frontend_status == 0:
            return 1, 1.0
        if frontend_latency > 1000.0:
            return 1, 0.8
        if frontend_latency > 700.0:
            return 1, 0.6
        return 0, 0.2


class BackendAgent:
    def suggest_action(self, state):
        backend_status = int(state[1])
        backend_latency = float(state[4])

        if backend_status == 0:
            return 2, 1.0
        if backend_latency > 1000.0:
            return 2, 0.8
        if backend_latency > 700.0:
            return 2, 0.6
        return 0, 0.2


class DatabaseAgent:
    def suggest_action(self, state):
        db_status = int(state[2])
        db_latency = float(state[5])

        if db_status == 0:
            return 3, 1.0
        if db_latency > 900.0:
            return 3, 0.8
        if db_latency > 650.0:
            return 3, 0.6
        return 0, 0.2


class TrafficAgent:
    def suggest_action(self, state):
        traffic_load = float(state[7])
        traffic_balance = float(state[8])
        request_queue = float(state[9])

        if request_queue > 300.0:
            return 5, 0.9
        if abs(traffic_balance - 0.5) > 0.25:
            return 5, 0.8
        if traffic_load > 0.85:
            return 4, 0.75
        return 0, 0.2


class CoordinatorAgent:
    def __init__(self):
        self.frontend_agent = FrontendAgent()
        self.backend_agent = BackendAgent()
        self.database_agent = DatabaseAgent()
        self.traffic_agent = TrafficAgent()

    def _rl_action(self, state, model):
        # RL receives the full environment state.
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def _is_critical(self, state):
        a_status = int(state[0])
        b_status = int(state[1])
        db_status = int(state[2])
        error_rate = float(state[6])
        return a_status == 0 or b_status == 0 or db_status == 0 or error_rate > 0.5

    def get_suggestions(self, state):
        frontend = self.frontend_agent.suggest_action(state)
        backend = self.backend_agent.suggest_action(state)
        database = self.database_agent.suggest_action(state)
        traffic = self.traffic_agent.suggest_action(state)
        return {
            "frontend": frontend,
            "backend": backend,
            "database": database,
            "traffic": traffic,
        }

    def select_action(self, state, model):
        action, _ = self.select_action_with_source(state, model)
        return action

    def select_action_with_source(self, state, model):
        # This is a hybrid system where rule-based safety overrides ensure reliability in critical states,
        # while RL handles general decision optimization.
        if not self._is_critical(state):
            return self._rl_action(state, model), "rl"

        db_status = int(state[2])
        b_status = int(state[1])
        a_status = int(state[0])
        error_rate = float(state[6])

        if db_status == 0:
            return 3, "rule:database"
        if b_status == 0:
            return 2, "rule:backend"
        if a_status == 0:
            return 1, "rule:frontend"
        if error_rate > 0.5:
            return 4, "rule:traffic"

        return self._rl_action(state, model), "rl"
