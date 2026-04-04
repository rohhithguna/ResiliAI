import random
from typing import Dict, Tuple


class SREEnvironment:
    """A minimal RL environment for autonomous incident response.

    State layout:
    [A_status, B_status, DB_status, A_latency, B_latency, DB_latency,
     global_error_rate, global_traffic_load, traffic_balance, request_queue]

    Status encoding:
    0 = down, 1 = degraded, 2 = healthy
    """

    STATUS_DOWN = 0
    STATUS_DEGRADED = 1
    STATUS_HEALTHY = 2

    ACTION_NOOP = 0
    ACTION_RESTART_A = 1
    ACTION_RESTART_B = 2
    ACTION_RESTART_DB = 3
    ACTION_THROTTLE_TRAFFIC = 4
    ACTION_REDISTRIBUTE_TRAFFIC = 5

    def __init__(self, max_steps: int = 40, seed: int = None):
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.a_status = self.STATUS_HEALTHY
        self.b_status = self.STATUS_HEALTHY
        self.db_status = self.STATUS_HEALTHY

        self.a_latency = 100.0
        self.b_latency = 120.0
        self.db_latency = 80.0

        self.global_error_rate = 0.0
        self.global_traffic_load = 0.5
        self.traffic_balance = 0.5
        self.request_queue = 100.0

        self.step_count = 0
        self.stable_steps = 0
        self.a_recovery_steps = 0
        self.b_recovery_steps = 0
        self.db_recovery_steps = 0
        self.traffic_mitigation = 0.0

    def _get_state(self) -> list:
        return [
            float(self.a_status),
            float(self.b_status),
            float(self.db_status),
            float(self.a_latency),
            float(self.b_latency),
            float(self.db_latency),
            float(self.global_error_rate),
            float(self.global_traffic_load),
            float(self.traffic_balance),
            float(self.request_queue),
        ]

    def extreme_scenario(self) -> str:
        """Inject a severe multi-component failure state for stress testing."""
        # Requested aliases for scenario readability.
        self.frontend_status = 1
        self.backend_status = 0
        self.db_status = 0
        self.traffic = 0.95

        self.a_status = self.STATUS_DEGRADED
        self.b_status = self.STATUS_DOWN
        self.db_status = self.STATUS_DOWN

        self.a_latency = max(self.a_latency, 900.0)
        self.b_latency = max(self.b_latency, 1600.0)
        self.db_latency = max(self.db_latency, 1800.0)

        self.global_error_rate = 0.9
        self.global_traffic_load = 0.95
        self.request_queue = 800.0
        self.traffic_balance = 0.5
        self.stable_steps = 0

        return "extreme_failure"

    def reset(self) -> list:
        """Resets the environment and injects initial failure patterns."""
        self.step_count = 0
        self.stable_steps = 0
        self.a_recovery_steps = 0
        self.b_recovery_steps = 0
        self.db_recovery_steps = 0
        self.traffic_mitigation = 0.0

        # Start from healthy baseline.
        self.a_status = self.STATUS_HEALTHY
        self.b_status = self.STATUS_HEALTHY
        self.db_status = self.STATUS_HEALTHY

        self.a_latency = self.rng.uniform(70.0, 130.0)
        self.b_latency = self.rng.uniform(90.0, 170.0)
        self.db_latency = self.rng.uniform(50.0, 120.0)

        self.global_error_rate = self.rng.uniform(0.0, 0.01)
        self.global_traffic_load = self.rng.uniform(0.3, 0.7)
        self.traffic_balance = self.rng.uniform(0.4, 0.6)
        self.request_queue = self.rng.uniform(50.0, 150.0)

        # Inject one failure and occasionally a second coupled failure.
        failure = self.rng.choice(["db_crash", "backend_overload", "traffic_surge"])

        if failure == "db_crash":
            self.db_status = self.STATUS_DOWN
            self.db_latency = 2000.0
            self.global_error_rate = max(self.global_error_rate, 0.30)
        elif failure == "backend_overload":
            self.b_status = self.STATUS_DEGRADED
            self.b_latency = self.rng.uniform(900.0, 1500.0)
            self.global_error_rate = max(self.global_error_rate, 0.12)
        else:  # traffic_surge
            self.global_traffic_load = self.rng.uniform(0.9, 1.0)
            self.a_latency += self.rng.uniform(120.0, 250.0)
            self.b_latency += self.rng.uniform(150.0, 300.0)
            self.db_latency += self.rng.uniform(80.0, 180.0)
            self.global_error_rate = max(self.global_error_rate, 0.05)

        if self.rng.random() < 0.15:
            extra = self.rng.choice(["frontend_down", "backend_down", "db_degraded"])
            if extra == "frontend_down":
                self.a_status = self.STATUS_DOWN
                self.a_latency = max(self.a_latency, 1500.0)
            elif extra == "backend_down":
                self.b_status = self.STATUS_DOWN
                self.b_latency = max(self.b_latency, 1500.0)
            else:
                self.db_status = min(self.db_status, self.STATUS_DEGRADED)
                self.db_latency = max(self.db_latency, 1000.0)

        if self.rng.random() < 0.05:
            # Rare overlap to force multi-step recovery policies.
            self.a_status = min(self.a_status, self.STATUS_DEGRADED)
            self.b_status = min(self.b_status, self.STATUS_DEGRADED)
            self.a_latency = max(self.a_latency, 900.0)
            self.b_latency = max(self.b_latency, 1100.0)

        self._update_error_rate()
        return self._get_state()

    def step(self, action: int) -> Tuple[list, float, bool, Dict[str, object]]:
        """Applies an action and advances the environment by one step.

        Returns:
            state, reward, done, info
        """
        self.step_count += 1

        # Track pre-action state to determine if this action improved the system.
        pre_healthy_services = sum(
            s == self.STATUS_HEALTHY for s in [self.a_status, self.b_status, self.db_status]
        )
        pre_error_rate = self.global_error_rate
        pre_total_latency = self.a_latency + self.b_latency + self.db_latency

        # 1) Apply action effects.
        if action == self.ACTION_RESTART_A:
            if self.a_status != self.STATUS_HEALTHY:
                self.a_status = self.STATUS_DEGRADED
                self.a_recovery_steps = self.rng.randint(2, 3)
            self.a_latency = min(self.a_latency, 420.0)
        elif action == self.ACTION_RESTART_B:
            if self.b_status != self.STATUS_HEALTHY:
                self.b_status = self.STATUS_DEGRADED
                self.b_recovery_steps = self.rng.randint(2, 3)
            self.b_latency = min(self.b_latency, 520.0)
        elif action == self.ACTION_RESTART_DB:
            if self.db_status != self.STATUS_HEALTHY:
                self.db_status = self.STATUS_DEGRADED
                self.db_recovery_steps = self.rng.randint(2, 3)
            self.db_latency = min(self.db_latency, 420.0)
        elif action == self.ACTION_THROTTLE_TRAFFIC:
            self.global_traffic_load = max(0.1, self.global_traffic_load - 0.15)
            self.traffic_mitigation = min(1.0, self.traffic_mitigation + 0.2)
        elif action == self.ACTION_REDISTRIBUTE_TRAFFIC:
            # Load balancer intervention.
            self.b_latency *= self.rng.uniform(0.8, 0.9)
            self.request_queue *= self.rng.uniform(0.6, 0.8)
            self.traffic_balance = 0.5 + (self.traffic_balance - 0.5) * 0.5
            self.traffic_mitigation = min(1.0, self.traffic_mitigation + 0.3)
        elif action != self.ACTION_NOOP:
            raise ValueError(f"Invalid action: {action}")

        # 2) System dynamics.
        # Traffic action effects persist for a few steps instead of being one-step artifacts.
        self.traffic_mitigation = max(0.0, self.traffic_mitigation - 0.05)

        if self.a_recovery_steps > 0:
            self.a_recovery_steps -= 1
            if self.a_recovery_steps == 0:
                self.a_status = self.STATUS_HEALTHY
        if self.b_recovery_steps > 0:
            self.b_recovery_steps -= 1
            if self.b_recovery_steps == 0:
                self.b_status = self.STATUS_HEALTHY
        if self.db_recovery_steps > 0:
            self.db_recovery_steps -= 1
            if self.db_recovery_steps == 0:
                self.db_status = self.STATUS_HEALTHY

        # Dependency propagation.
        if self.db_status == self.STATUS_DOWN:
            self.b_status = min(self.b_status, self.STATUS_DEGRADED)
            self.b_latency = max(self.b_latency, 900.0)

        if self.b_status == self.STATUS_DOWN:
            self.a_status = min(self.a_status, self.STATUS_DEGRADED)
            self.a_latency = max(self.a_latency, 800.0)

        # High traffic increases latency.
        if self.global_traffic_load > 0.8:
            self.a_latency += 150.0
            self.b_latency += 180.0
            self.db_latency += 120.0
        elif self.global_traffic_load > 0.6:
            self.a_latency += 60.0
            self.b_latency += 70.0
            self.db_latency += 50.0
        else:
            # Mild decay toward normal when traffic is moderate/low.
            self.a_latency = max(60.0, self.a_latency - 30.0)
            self.b_latency = max(80.0, self.b_latency - 35.0)
            self.db_latency = max(40.0, self.db_latency - 25.0)

        # Queue dynamics.
        if self.global_traffic_load > 0.8:
            self.request_queue += self.rng.uniform(20.0, 80.0)
        if self.b_latency > 900.0:
            self.request_queue += self.rng.uniform(10.0, 50.0)

        # Traffic balancing dynamics.
        self.traffic_balance += self.rng.uniform(-0.03, 0.03)
        self.traffic_balance = max(0.0, min(1.0, self.traffic_balance))
        imbalance = abs(self.traffic_balance - 0.5)
        if imbalance > 0.2:
            self.b_latency += 80.0 * imbalance

        if self.request_queue > 300.0:
            queue_pressure = min(1.0, (self.request_queue - 300.0) / 700.0)
            self.global_error_rate += 0.05 * queue_pressure
            self.a_latency += 60.0 * queue_pressure
            self.b_latency += 80.0 * queue_pressure
            self.db_latency += 40.0 * queue_pressure

        # Latency overflow can cause degraded status.
        if self.a_latency > 1200.0 and self.a_status == self.STATUS_HEALTHY:
            self.a_status = self.STATUS_DEGRADED
        if self.b_latency > 1200.0 and self.b_status == self.STATUS_HEALTHY:
            self.b_status = self.STATUS_DEGRADED
        if self.db_latency > 1200.0 and self.db_status == self.STATUS_HEALTHY:
            self.db_status = self.STATUS_DEGRADED

        # Clamp latency bounds.
        self.a_latency = max(0.0, min(2000.0, self.a_latency))
        self.b_latency = max(0.0, min(2000.0, self.b_latency))
        self.db_latency = max(0.0, min(2000.0, self.db_latency))
        self.request_queue = max(0.0, min(1000.0, self.request_queue))

        # 3) Update error rate from health and latency.
        self._update_error_rate()

        # 4) Done conditions.
        is_stable_now = (
            self.a_status == self.STATUS_HEALTHY
            and self.b_status == self.STATUS_HEALTHY
            and self.db_status == self.STATUS_HEALTHY
            and self.global_error_rate < 0.02
        )
        if is_stable_now:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        success = self.stable_steps >= 2
        failure = self.global_error_rate > 0.6
        timeout = self.step_count >= self.max_steps

        done = success or failure or timeout

        # 5) Compute simple, high-impact reward.
        post_healthy_services = sum(
            s == self.STATUS_HEALTHY for s in [self.a_status, self.b_status, self.db_status]
        )
        improved = (
            post_healthy_services > pre_healthy_services
            or self.global_error_rate < (pre_error_rate - 1e-4)
        )

        post_total_latency = self.a_latency + self.b_latency + self.db_latency
        error_delta = pre_error_rate - self.global_error_rate
        latency_delta = pre_total_latency - post_total_latency

        reward = 0.0
        if success:
            reward += 10.0
        reward += -12.0 * self.global_error_rate
        reward += -0.004 * post_total_latency
        reward += -0.5

        # Small, capped progress bonus to encourage stepwise recovery without reward hacking.
        reward += max(-1.5, min(1.5, 6.0 * error_delta + 0.0008 * latency_delta))

        # Penalize unnecessary actions when system is already near-stable.
        if action != self.ACTION_NOOP and pre_error_rate < 0.03 and pre_total_latency < 500.0:
            reward -= 0.8

        # Penalize dead-end terminations.
        if done and not success:
            reward -= 3.0

        info = {
            "success": success,
            "failure": failure,
            "timeout": timeout,
            "step_count": self.step_count,
        }

        return self._get_state(), float(reward), done, info

    def _update_error_rate(self) -> None:
        error = 0.0

        for status in [self.a_status, self.b_status, self.db_status]:
            if status == self.STATUS_DOWN:
                error += 0.25
            elif status == self.STATUS_DEGRADED:
                error += 0.08

        total_latency = self.a_latency + self.b_latency + self.db_latency
        error += max(0.0, (total_latency - 450.0) / 5000.0)
        error += max(0.0, self.global_traffic_load - 0.7) * 0.2

        # Persistent mitigation effect from traffic actions.
        error -= 0.08 * self.traffic_mitigation

        self.global_error_rate = max(0.0, min(1.0, error))
