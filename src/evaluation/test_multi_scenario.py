"""
STAGE 4: Multi-Scenario Testing
Comprehensive testing across multiple scenarios with different policies.
"""

import sys
sys.path.insert(0, '/Users/rohhithg/Desktop/meta_project')

from sre_openenv import SREOpenEnv
from inference import select_action as ai_select_action
from metrics_and_baseline import SimpleRulePolicy, MetricsTracker


class MultiScenarioTester:
    """Run comprehensive tests across multiple scenarios and policies."""
    
    def __init__(self, max_steps: int = 40):
        self.max_steps = max_steps
        self.results = {}
    
    def run_multiple_tests(self, scenarios: list, policy_fn, policy_name: str, 
                          runs: int = 5, seed_base: int = 42) -> dict:
        """Run multiple tests across scenarios with a given policy.
        
        Args:
            scenarios: List of scenario names to test
            policy_fn: Function that takes state and returns action
            policy_name: Name of the policy (for tracking)
            runs: Number of runs per scenario
            seed_base: Base seed for reproducibility
        
        Returns:
            Dictionary with statistics per scenario
        """
        print(f"\n{'='*70}")
        print(f"Testing Policy: {policy_name}")
        print(f"{'='*70}")
        
        stats = {}
        
        for scenario in scenarios:
            print(f"\n[{scenario.upper()}] - {runs} runs)")
            
            steps_list = []
            errors_list = []
            recovered_count = 0
            rewards_list = []
            
            for run in range(runs):
                env = SREOpenEnv(seed=seed_base + run)
                state = env.reset()
                env.inject_scenario(scenario)
                
                state = env.state()
                initial_error = state["error_rate"]
                steps = 0
                total_reward = 0
                
                while steps < self.max_steps:
                    action = policy_fn(state)
                    state, reward, done, _ = env.step(action)
                    steps += 1
                    total_reward += reward
                    
                    # Check recovery condition
                    if done and state.get("error_rate", 1.0) < 0.05:
                        recovered_count += 1
                        break
                
                final_error = state.get("error_rate", 1.0)
                steps_list.append(steps)
                errors_list.append(final_error)
                rewards_list.append(total_reward)
            
            avg_steps = sum(steps_list) / len(steps_list)
            avg_error = sum(errors_list) / len(errors_list)
            avg_reward = sum(rewards_list) / len(rewards_list)
            recovery_rate = recovered_count / runs
            
            stats[scenario] = {
                "avg_steps": avg_steps,
                "avg_error": avg_error,
                "avg_reward": avg_reward,
                "recovery_rate": recovery_rate,
                "steps_list": steps_list,
                "errors_list": errors_list
            }
            
            print(f"  Avg Steps: {avg_steps:5.1f} | Avg Error: {avg_error:.3f} | "
                  f"Recovery: {recovery_rate:.0%} | Reward: {avg_reward:+7.2f}")
        
        self.results[policy_name] = stats
        return stats
    
    def compare_policies(self):
        """Compare all tested policies."""
        if len(self.results) < 2:
            print("\nNeed at least 2 policies to compare")
            return
        
        print(f"\n{'='*70}")
        print("POLICY COMPARISON")
        print(f"{'='*70}")
        
        all_scenarios = set()
        for policy_stats in self.results.values():
            all_scenarios.update(policy_stats.keys())
        
        for scenario in sorted(all_scenarios):
            print(f"\n[{scenario.upper()}]")
            
            policies_sorted = []
            for policy_name, scenario_stats in self.results.items():
                if scenario in scenario_stats:
                    stats = scenario_stats[scenario]
                    policies_sorted.append((
                        policy_name,
                        stats["avg_error"],
                        stats["recovery_rate"],
                        stats["avg_steps"]
                    ))
            
            # Sort by error (lower is better)
            policies_sorted.sort(key=lambda x: x[1])
            
            for i, (policy_name, avg_error, recovery_rate, avg_steps) in enumerate(policies_sorted):
                marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"  {marker} {policy_name:15s}: error={avg_error:.3f}, "
                      f"recovery={recovery_rate:.0%}, steps={avg_steps:5.1f}")
    
    def summary_table(self):
        """Print a summary table of all results."""
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}\n")
        
        # Collect all scenarios
        all_scenarios = set()
        for policy_stats in self.results.values():
            all_scenarios.update(policy_stats.keys())
        
        # Print header
        header = "Policy".ljust(20) + "".join(f"{s:15s}" for s in sorted(all_scenarios))
        print(header)
        print("-" * len(header))
        
        # Print rows (recovery rate)
        for policy_name in sorted(self.results.keys()):
            row = policy_name.ljust(20)
            for scenario in sorted(all_scenarios):
                if scenario in self.results[policy_name]:
                    rate = self.results[policy_name][scenario]["recovery_rate"]
                    row += f"{rate:14.0%} "
                else:
                    row += "     N/A        "
            print(row)


def test_multi_scenario():
    """Run comprehensive multi-scenario tests."""
    
    tester = MultiScenarioTester(max_steps=20)
    
    scenarios = ["traffic_spike", "db_failure", "multi_failure"]
    
    # Test AI policy
    tester.run_multiple_tests(
        scenarios=scenarios,
        policy_fn=lambda state: ai_select_action(state),
        policy_name="AI Policy",
        runs=5
    )
    
    # Test baseline policy
    tester.run_multiple_tests(
        scenarios=scenarios,
        policy_fn=lambda state: SimpleRulePolicy.select_action(state),
        policy_name="Baseline (Rules)",
        runs=5
    )
    
    # Compare results
    tester.compare_policies()
    tester.summary_table()
    
    print(f"\n{'='*70}")
    print("✅ Stage 4 multi-scenario testing complete!")
    print(f"{'='*70}")
    
    return tester


if __name__ == "__main__":
    print("="*70)
    print("STAGE 4: MULTI-SCENARIO TESTING")
    print("="*70)
    
    tester = test_multi_scenario()
