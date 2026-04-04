from sre_openenv import SREOpenEnv
from inference import select_action, _state_to_obs

env = SREOpenEnv(seed=42)
state = env.reset()

print("===== DEMO START =====")

for step in range(10):
    action = select_action(state)
    state, reward, done, _ = env.step(action)

    obs = _state_to_obs(state)

    print(f"\nStep {step+1}")
    print(f"Action: {action}")
    print(f"Frontend: {obs['frontend_status']} | Backend: {obs['backend_status']} | DB: {obs['db_status']}")
    print(f"Error: {obs['error_rate']:.3f} | Traffic: {obs['traffic_load']:.3f}")
    print(f"Queue: {obs['request_queue']:.1f}")

    if done:
        print("Recovered or terminated")
        break

print("\n===== DEMO END =====")
