import os
import sys
import time

import streamlit as st
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from multi_agent import CoordinatorAgent
from sre_environment import SREEnvironment
from train_dqn import QNetwork


STATE_DIM = 10
ACTION_DIM = 6
MODEL_PATH = os.path.join(ROOT_DIR, "dqn_model.pth")

ACTION_TEXT = {
    0: "No action",
    1: "Restart Frontend",
    2: "Restart Backend",
    3: "Restart Database",
    4: "Throttle Traffic",
    5: "Redistribute Traffic",
}

STATUS_TEXT = {
    0: "🔴 Down",
    1: "🟡 Degraded",
    2: "🟢 Healthy",
}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run train_dqn.py first")

    model = QNetwork(STATE_DIM, ACTION_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def init_simulation():
    env = SREEnvironment(seed=42)
    st.session_state.env = env
    st.session_state.state = env.reset()
    st.session_state.step_count = 0
    st.session_state.done = False
    st.session_state.last_action = None
    st.session_state.last_reason = "Simulation initialized"
    st.session_state.last_result = "-"
    st.session_state.log = []
    st.session_state.suggestions = {}


def get_reason(state):
    a, b, db = int(state[0]), int(state[1]), int(state[2])
    err = float(state[6])

    if db == 0:
        return "DB down -> restarting database"
    if b == 0:
        return "Backend issue -> restarting backend"
    if a == 0:
        return "Frontend issue -> restarting frontend"
    if err > 0.3:
        return "High error -> throttling traffic"
    return "System stable"


def health_summary(state):
    h = sum(int(x) == 2 for x in state[:3])
    if h == 3:
        return "Healthy"
    if h == 2:
        return "Warning"
    return "Critical"


def recovery_status(done, env):
    if done and env.global_error_rate < 0.02:
        return "Recovered"
    if done:
        return "Not recovered"
    return "In progress"


def run_one_step(model, coordinator):
    if st.session_state.env is None:
        init_simulation()
    if st.session_state.done:
        return

    state = st.session_state.state
    suggestions = coordinator.get_suggestions(state)
    action = coordinator.select_action(state, model)
    next_state, reward, done, _ = st.session_state.env.step(action)

    issue = get_reason(state)
    if done and st.session_state.env.global_error_rate < 0.02:
        result = "Recovered"
    elif done:
        result = "Episode ended"
    elif reward > 0:
        result = "Improving"
    else:
        result = "Needs attention"

    st.session_state.state = next_state
    st.session_state.step_count += 1
    st.session_state.done = done
    st.session_state.last_action = action
    st.session_state.last_reason = issue
    st.session_state.last_result = result
    st.session_state.suggestions = suggestions

    st.session_state.log.append(
        {
            "Step": st.session_state.step_count,
            "Issue Detected": issue,
            "Action Taken": ACTION_TEXT.get(action, str(action)),
            "Result": result,
        }
    )
    st.session_state.log = st.session_state.log[-5:]


st.set_page_config(page_title="AI SRE Decision System", layout="wide")
st.title("AI SRE Decision System")
st.caption("Multi-Agent AI System for Autonomous Incident Recovery")

try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.state = None
    st.session_state.step_count = 0
    st.session_state.done = False
    st.session_state.last_action = None
    st.session_state.last_reason = ""
    st.session_state.last_result = "-"
    st.session_state.log = []
    st.session_state.suggestions = {}

coordinator = CoordinatorAgent()

st.subheader("Controls")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Start", use_container_width=True):
        init_simulation()

with col2:
    if st.button("Next Step", use_container_width=True):
        run_one_step(model, coordinator)

with col3:
    if st.button("Run Auto Demo", use_container_width=True):
        if st.session_state.env is None:
            init_simulation()
        for _ in range(10):
            if st.session_state.done:
                break
            run_one_step(model, coordinator)
            time.sleep(0.12)
        st.rerun()

with col4:
    if st.button("Reset", use_container_width=True):
        init_simulation()

if st.session_state.state is None:
    st.info("Press Start to begin simulation.")
    st.stop()

state = st.session_state.state
env = st.session_state.env

if any(int(state[i]) == 0 for i in [0, 1, 2]):
    st.warning("Critical state detected: at least one core service is down.")

st.subheader("System State")
st.table(
    {
        "Component": ["Frontend", "Backend", "DB"],
        "Status": [
            STATUS_TEXT[int(state[0])],
            STATUS_TEXT[int(state[1])],
            STATUS_TEXT[int(state[2])],
        ],
        "Latency": [
            f"{state[3]:.1f} ms",
            f"{state[4]:.1f} ms",
            f"{state[5]:.1f} ms",
        ],
    }
)

st.subheader("Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("System Health", health_summary(state))
m2.metric("Error Rate", f"{state[6]:.4f}")
m3.metric("Traffic Load", f"{state[7]:.4f}")
m4.metric("Queue Length", f"{state[9]:.1f}")
m5.metric("Step Count", st.session_state.step_count)

st.subheader("Agent Decisions")
suggestions = st.session_state.suggestions or coordinator.get_suggestions(state)

def suggestion_text(name):
    action, conf = suggestions[name]
    return f"{ACTION_TEXT.get(action, str(action))} (conf={conf:.2f})"

d1, d2 = st.columns(2)
with d1:
    st.write(f"Frontend Agent -> {suggestion_text('frontend')}")
    st.write(f"Backend Agent -> {suggestion_text('backend')}")
with d2:
    st.write(f"DB Agent -> {suggestion_text('database')}")
    st.write(f"Traffic Agent -> {suggestion_text('traffic')}")

st.write(f"Final Decision -> {ACTION_TEXT.get(st.session_state.last_action, '-')}")
st.write(f"Reason -> {st.session_state.last_reason}")

st.subheader("Recovery Status")
st.write(recovery_status(st.session_state.done, env))
if st.session_state.done:
    if env.global_error_rate < 0.02:
        st.success("System recovered.")
    else:
        st.warning("Episode finished without full recovery.")

st.subheader("Recent Actions")
if st.session_state.log:
    st.table(st.session_state.log)
else:
    st.write("No actions yet.")
