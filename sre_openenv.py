ACTIONS = {
    0: "noop",
    1: "restart_frontend",
    2: "restart_backend",
    3: "restart_database",
    4: "throttle_traffic",
    5: "rebalance_load"
}


class SREOpenEnv:
    def __init__(self, seed=None):
        from sre_environment import SREEnvironment
        self.env = SREEnvironment(seed=seed)
        self.step_count = 0
        if not hasattr(self.env, "current_state"):
            self.env.current_state = self.env._get_state()

    def reset(self):
        state = self.env.reset()
        self.step_count = 0
        self.env.current_state = state
        return self._format_obs(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.step_count += 1
        self.env.current_state = next_state

        return (
            self._format_obs(next_state),
            float(reward),
            bool(done),
            self._format_info(next_state, done)
        )

    def state(self):
        if hasattr(self.env, "current_state"):
            return self._format_obs(self.env.current_state)
        else:
            return self._format_obs(self.env._get_state() if hasattr(self.env, "_get_state") else self.env.reset())

    def _format_obs(self, state):
        return {
            "frontend_status": int(state[0]),
            "backend_status": int(state[1]),
            "db_status": int(state[2]),
            "frontend_latency": float(state[3]),
            "backend_latency": float(state[4]),
            "db_latency": float(state[5]),
            "error_rate": float(state[6]),
            "traffic_load": float(state[7]),
            "traffic_balance": float(state[8]),
            "request_queue": float(state[9])
        }

    def _format_info(self, state, done):
        return {
            "health_score": self._health_score(state),
            "recovered": bool(done),
            "step": int(self.step_count)
        }

    def _health_score(self, state):
        healthy = sum(int(s) == 2 for s in state[:3])
        return healthy / 3.0


if __name__ == "__main__":
    print("=== OpenEnv Validation Start ===")

    env = SREOpenEnv(seed=42)

    print("\n[RESET TEST]")
    obs = env.reset()
    print(obs)

    print("\n[STEP TEST]")
    for i in range(5):
        obs, reward, done, info = env.step(0)
        print(f"\nStep {i+1}")
        print("Obs:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

    print("\n[STATE TEST]")
    print(env.state())

    print("\n=== VALIDATION COMPLETE ===")
