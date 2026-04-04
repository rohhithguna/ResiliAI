"""
STAGE 1: Realistic Failure Scenarios
Generates and tests different incident scenarios for the AutoRecovery system.
"""

import sys
sys.path.insert(0, '/Users/rohhithg/Desktop/meta_project')

from sre_openenv import SREOpenEnv
from inference import select_action

def test_scenario(scenario_name: str, num_steps: int = 10):
    """Test how the AI handles a specific scenario."""
    print(f"\n{'='*60}")
    print(f"Testing Scenario: {scenario_name.upper()}")
    print(f"{'='*60}")
    
    env = SREOpenEnv(seed=42)
    env.reset()
    env.inject_scenario(scenario_name)
    
    state = env.state()
    print(f"\nInitial State:")
    print(f"  Frontend: {state['frontend_status']:1.0f}, Backend: {state['backend_status']:1.0f}, DB: {state['db_status']:1.0f}")
    print(f"  Error Rate: {state['error_rate']:.3f}, Traffic: {state['traffic_load']:.3f}")
    
    total_reward = 0
    actions_taken = []
    
    print(f"\nRecovery Actions (up to {num_steps} steps):")
    for step in range(num_steps):
        action = select_action(state)
        actions_taken.append(action)
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        
        action_name = ["NOOP", "Frontend↻", "Backend↻", "DB↻", "Throttle", "Rebalance"][action]
        print(f"  Step {step+1}: Action={action_name:12s} | Reward={reward:+6.2f} | Error={new_state['error_rate']:.3f}")
        
        state = new_state
        if done and state['error_rate'] < 0.05:
            print(f"\n✅ RECOVERED in {step+1} steps!")
            break
    
    if not (done and state['error_rate'] < 0.05):
        print(f"\n⚠️  Not recovered after {num_steps} steps (error: {state['error_rate']:.3f})")
    
    print(f"\nSummary:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Actions: {actions_taken}")
    print(f"  Final Error Rate: {state['error_rate']:.3f}")
    
    return {
        "scenario": scenario_name,
        "steps": len(actions_taken),
        "recovered": done and state['error_rate'] < 0.05,
        "reward": total_reward,
        "error_rate": state['error_rate'],
        "actions": actions_taken
    }

if __name__ == "__main__":
    print("="*60)
    print("STAGE 1: REALISTIC FAILURE SCENARIOS")
    print("="*60)
    
    results = []
    for scenario in ["traffic_spike", "db_failure", "multi_failure"]:
        result = test_scenario(scenario, num_steps=15)
        results.append(result)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    for result in results:
        status = "✅ PASS" if result["recovered"] else "⚠️  PARTIAL"
        print(f"{result['scenario']:15s}: {status} | Steps: {result['steps']:2d} | Error: {result['error_rate']:.3f} | Reward: {result['reward']:+7.2f}")
    
    print(f"\n✅ Stage 1 validation complete!")
