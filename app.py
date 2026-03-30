import os
import sys

import streamlit as st
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sre_environment import SREEnvironment
from train_dqn import QNetwork


STATE_DIM = 10
ACTION_DIM = 6
MODEL_PATH = os.path.join(ROOT_DIR, "dqn_model.pth")

ACTION_NAMES = {
    0: "No action",
    1: "Restart Frontend",
    2: "Restart Backend",
    3: "Restart Database",
    4: "Throttle Traffic",
    5: "Redistribute Traffic",
}

STATUS_NAMES = {
    0: "Down",
    1: "Degraded",
    2: "Healthy",
}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_dqn.py first.")

    model = QNetwork(STATE_DIM, ACTION_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def initialize_session_state():
    if "env" not in st.session_state:
        st.session_state.env = None
    if "state" not in st.session_state:
        st.session_state.state = None
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "done" not in st.session_state:
        st.session_state.done = False
    if "last_action" not in st.session_state:
        st.session_state.last_action = None
    if "log" not in st.session_state:
        st.session_state.log = []


def init_simulation():
    env = SREEnvironment(seed=42)
    state = env.reset()

    st.session_state.env = env
    st.session_state.state = state
    st.session_state.step_count = 0
    st.session_state.done = False
    st.session_state.last_action = None
    st.session_state.log = []


def select_rl_action(model, state):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def next_step(model):
    if st.session_state.env is None or st.session_state.state is None:
        init_simulation()

    if st.session_state.done:
        return

    current_state = st.session_state.state
    action = select_rl_action(model, current_state)

    next_state, _, done, _ = st.session_state.env.step(action)

    st.session_state.state = next_state
    st.session_state.done = done
    st.session_state.step_count += 1
    st.session_state.last_action = action

    st.session_state.log.append(
        {
            "Step": st.session_state.step_count,
            "Action": ACTION_NAMES.get(action, str(action)),
            "Error Rate": f"{next_state[6]:.4f}",
        }
    )
    st.session_state.log = st.session_state.log[-5:]


def recovery_status():
    if st.session_state.env is None:
        return "Not started"
    if not st.session_state.done:
        return "In progress"
    if st.session_state.env.global_error_rate < 0.02:
        return "Recovered"
    return "Failed / Timeout"


st.set_page_config(page_title="AI SRE System", layout="wide")
st.title("AI SRE Decision System")

initialize_session_state()

try:
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

col_start, col_next, col_reset = st.columns(3)

with col_start:
    if st.button("Start", use_container_width=True):
        init_simulation()

with col_next:
    if st.button("Next Step", use_container_width=True):
        next_step(model)

with col_reset:
    if st.button("Reset", use_container_width=True):
        init_simulation()

if st.session_state.state is None:
    st.info("Click Start to begin simulation.")
    st.stop()

state = st.session_state.state

st.subheader("System Components")
st.table(
    {
        "Component": ["Frontend", "Backend", "Database"],
        "Status": [
            STATUS_NAMES.get(int(state[0]), "Unknown"),
            STATUS_NAMES.get(int(state[1]), "Unknown"),
            STATUS_NAMES.get(int(state[2]), "Unknown"),
        ],
        "Latency (ms)": [
            f"{state[3]:.1f}",
            f"{state[4]:.1f}",
            f"{state[5]:.1f}",
        ],
    }
)

st.subheader("System Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Error Rate", f"{state[6]:.4f}")
m2.metric("Traffic Load", f"{state[7]:.4f}")
m3.metric("Traffic Balance", f"{state[8]:.4f}")
m4.metric("Queue", f"{state[9]:.1f}")
m5.metric("Step", st.session_state.step_count)

st.subheader("Decision")
st.write(f"Last Action: {ACTION_NAMES.get(st.session_state.last_action, '-')}")
st.write(f"Recovery Status: {recovery_status()}")

if st.session_state.done:
    if st.session_state.env.global_error_rate < 0.02:
        st.success("Episode completed: system recovered.")
    else:
        st.warning("Episode completed: recovery not achieved.")

st.subheader("Recent Action Log")
if st.session_state.log:
    st.table(st.session_state.log)
else:
    st.write("No actions yet.")
