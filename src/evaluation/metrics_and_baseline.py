"""
STAGE 2: Metrics Tracking
Track performance metrics for recovery operations.

STAGE 3: Baseline Comparison
Simple rule-based policy for baseline comparison against RL agent.
"""

import sys
sys.path.insert(0, '/Users/rohhithg/Desktop/meta_project')


# ========================================
# STAGE 2: METRICS TRACKING
# ========================================

class MetricsTracker:
    """Track and compute recovery metrics."""
    
    def __init__(self):
        self.episodes = []
    
    def log_episode(self, initial_error: float, final_error: float, 
                    steps: int, actions: list, reward: float, scenario: str):
        """Log a single recovery episode."""
        metrics = self.compute_metrics(initial_error, final_error, steps)
        
        episode = {
            "scenario": scenario,
            "initial_error": initial_error,
            "final_error": final_error,
            "improvement": metrics["improvement"],
            "steps": steps,
            "efficiency": metrics["efficiency"],
            "actions": actions,
            "reward": reward,
            "recovered": final_error < 0.05
        }
        
        self.episodes.append(episode)
        return episode
    
    @staticmethod
    def compute_metrics(initial_error: float, final_error: float, steps: int) -> dict:
        """Compute improvement and efficiency metrics."""
        improvement = initial_error - final_error
        efficiency = improvement / max(steps, 1) if steps > 0 else 0.0
        
        return {
            "initial_error": initial_error,
            "final_error": final_error,
            "improvement": improvement,
            "steps": steps,
            "efficiency": efficiency
        }
    
    def summary(self) -> dict:
        """Compute overall summary statistics."""
        if not self.episodes:
            return {}
        
        total_episodes = len(self.episodes)
        recovered = sum(1 for e in self.episodes if e["recovered"])
        
        avg_steps = sum(e["steps"] for e in self.episodes) / total_episodes
        avg_efficiency = sum(e["efficiency"] for e in self.episodes) / total_episodes
        avg_improvement = sum(e["improvement"] for e in self.episodes) / total_episodes
        
        return {
            "total_episodes": total_episodes,
            "recovered_count": recovered,
            "recovery_rate": recovered / total_episodes,
            "avg_steps": avg_steps,
            "avg_efficiency": avg_efficiency,
            "avg_improvement": avg_improvement
        }


# ========================================
# STAGE 3: BASELINE COMPARISON
# ========================================

class SimpleRulePolicy:
    """Simple rule-based policy for baseline comparison."""
    
    @staticmethod
    def select_action(state) -> int:
        """Select action using simple prioritized rules.
        
        Priority order:
        1. Database down → restart (3)
        2. Backend down → restart (2)
        3. Frontend down → restart (1)
        4. High error rate → throttle (4)
        5. Default → no-op (0)
        """
        # Handle dict-style state
        if isinstance(state, dict):
            db_status = state.get("db_status", 2)
            backend_status = state.get("backend_status", 2)
            frontend_status = state.get("frontend_status", 2)
            error_rate = state.get("error_rate", 0.0)
        else:
            # Handle list-style state
            db_status = int(state[2])
            backend_status = int(state[1])
            frontend_status = int(state[0])
            error_rate = float(state[6])
        
        # Priority-based decision tree
        if db_status == 0:
            return 3  # Restart database (highest priority)
        elif backend_status == 0:
            return 2  # Restart backend
        elif frontend_status == 0:
            return 1  # Restart frontend
        elif error_rate > 0.3:
            return 4  # Throttle traffic on high error
        else:
            return 0  # No-op, system ok


def test_baseline_vs_ai():
    """Compare baseline rule policy against AI policies."""
    from sre_openenv import SREOpenEnv
    from inference import select_action as ai_select_action
    
    print("\n" + "="*70)
    print("BASELINE vs AI COMPARISON")
    print("="*70)
    
    scenarios = ["traffic_spike", "db_failure", "multi_failure"]
    tracker_baseline = MetricsTracker()
    tracker_ai = MetricsTracker()
    
    for scenario in scenarios:
        print(f"\n[{scenario.upper()}]")
        
        # Test baseline
        env = SREOpenEnv(seed=42)
        env.reset()
        env.inject_scenario(scenario)
        
        state = env.state()
        initial_error = state["error_rate"]
        actions_baseline = []
        reward_baseline = 0
        
        for step in range(15):
            if isinstance(state, dict):
                list_state = [state.get(k) for k in [
                    "frontend_status", "backend_status", "db_status",
                    "frontend_latency", "backend_latency", "db_latency",
                    "error_rate", "traffic_load", "traffic_balance", "request_queue"
                ]]
            else:
                list_state = state
            
            action = SimpleRulePolicy.select_action(list_state)
            actions_baseline.append(action)
            state, reward, done, _ = env.step(action)
            reward_baseline += reward
        
        final_error_baseline = state["error_rate"]
        tracker_baseline.log_episode(initial_error, final_error_baseline, 15, 
                                      actions_baseline, reward_baseline, scenario)
        
        # Test AI
        env = SREOpenEnv(seed=42)
        env.reset()
        env.inject_scenario(scenario)
        
        state = env.state()
        initial_error = state["error_rate"]
        actions_ai = []
        reward_ai = 0
        
        for step in range(15):
            action = ai_select_action(state)
            actions_ai.append(action)
            state, reward, done, _ = env.step(action)
            reward_ai += reward
        
        final_error_ai = state["error_rate"]
        tracker_ai.log_episode(initial_error, final_error_ai, 15, 
                                actions_ai, reward_ai, scenario)
        
        # Print comparison
        print(f"  Baseline: error {initial_error:.3f} → {final_error_baseline:.3f} " 
              f"| reward: {reward_baseline:+.2f}")
        print(f"  AI:       error {initial_error:.3f} → {final_error_ai:.3f} " 
              f"| reward: {reward_ai:+.2f}")
        
        improvement_baseline = initial_error - final_error_baseline
        improvement_ai = initial_error - final_error_ai
        
        if improvement_ai > improvement_baseline:
            print(f"  ✅ AI BETTER (+{improvement_ai - improvement_baseline:.3f} improvement)")
        else:
            print(f"  ⚠️  Baseline keeps up")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary_baseline = tracker_baseline.summary()
    summary_ai = tracker_ai.summary()
    
    print(f"\nBaseline Policy:")
    print(f"  Recovery Rate: {summary_baseline['recovery_rate']:.1%}")
    print(f"  Avg Steps: {summary_baseline['avg_steps']:.1f}")
    print(f"  Avg Efficiency: {summary_baseline['avg_efficiency']:.3f}")
    print(f"  Avg Improvement: {summary_baseline['avg_improvement']:.3f}")
    
    print(f"\nAI Policy:")
    print(f"  Recovery Rate: {summary_ai['recovery_rate']:.1%}")
    print(f"  Avg Steps: {summary_ai['avg_steps']:.1f}")
    print(f"  Avg Efficiency: {summary_ai['avg_efficiency']:.3f}")
    print(f"  Avg Improvement: {summary_ai['avg_improvement']:.3f}")
    
    print(f"\n✅ Stage 2 & 3 validation complete!")


if __name__ == "__main__":
    print("="*70)
    print("STAGE 2: METRICS TRACKING")
    print("="*70)
    
    tracker = MetricsTracker()
    
    # Example: log some metrics
    metrics = tracker.compute_metrics(initial_error=0.6, final_error=0.05, steps=10)
    print(f"\nMetrics Example:")
    print(f"  Initial Error: {metrics['initial_error']:.3f}")
    print(f"  Final Error: {metrics['final_error']:.3f}")
    print(f"  Improvement: {metrics['improvement']:.3f}")
    print(f"  Efficiency: {metrics['efficiency']:.3f}")
    
    print(f"\n{'='*70}")
    print(f"STAGE 3: BASELINE COMPARISON")
    print(f"{'='*70}")
    
    test_baseline_vs_ai()
