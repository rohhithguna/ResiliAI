from typing import Sequence


class RuleAgent:
    """Simple rule-based policy for SREEnvironment.

    State format:
    [A_status, B_status, DB_status, A_latency, B_latency, DB_latency, error_rate, traffic]
    """

    def select_action(self, state: Sequence[float]) -> int:
        """Select an action using fixed-priority incident-response rules."""
        if len(state) < 8:
            raise ValueError("State must contain 8 elements.")

        a_status = state[0]
        b_status = state[1]
        db_status = state[2]
        error_rate = state[6]

        if db_status == 0:
            print("[WARN] Database failure -> Restart Database")
            return 3  # Restart DB
        if b_status == 0:
            return 2  # Restart B
        if a_status == 0:
            return 1  # Restart A
        if error_rate > 0.3:
            print("[WARN] Traffic issue -> Throttle Traffic")
            return 4  # Throttle traffic
        return 0  # No-op
