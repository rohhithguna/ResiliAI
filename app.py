import streamlit as st
from sre_openenv import SREOpenEnv
from inference import select_action

st.set_page_config(page_title="AI SRE System", layout="wide")

st.title("AI System for Autonomous Incident Recovery")

# --------------------------------
# INIT STATE
# --------------------------------
if "env" not in st.session_state:
    st.session_state.env = SREOpenEnv(seed=42)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.log = []

# --------------------------------
# CONTROLS
# --------------------------------
st.subheader("Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start / Reset"):
        st.session_state.env = SREOpenEnv(seed=42)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.step = 0
        st.session_state.done = False
        st.session_state.log = []

with col2:
    if st.button("Next Step"):
        if not st.session_state.done:
            action = select_action(st.session_state.state)
            new_state, reward, done, _ = st.session_state.env.step(action)

            st.session_state.log.append({
                "step": st.session_state.step,
                "action": action,
                "reward": reward
            })

            st.session_state.state = new_state
            st.session_state.done = done
            st.session_state.step += 1

with col3:
    if st.button("Auto Run (10 steps)"):
        for _ in range(10):
            if not st.session_state.done:
                action = select_action(st.session_state.state)
                new_state, reward, done, _ = st.session_state.env.step(action)

                st.session_state.log.append({
                    "step": st.session_state.step,
                    "action": action,
                    "reward": reward
                })

                st.session_state.state = new_state
                st.session_state.done = done
                st.session_state.step += 1

# --------------------------------
# DISPLAY STATE
# --------------------------------
state = st.session_state.state

st.subheader("System State")

try:
    frontend, backend, db = int(state[0]), int(state[1]), int(state[2])
    fl, bl, dl = state[3], state[4], state[5]
    err, traffic = state[6], state[7]
except Exception:
    frontend = state.get("frontend_status", 0)
    backend = state.get("backend_status", 0)
    db = state.get("db_status", 0)
    fl = state.get("frontend_latency", 0)
    bl = state.get("backend_latency", 0)
    dl = state.get("db_latency", 0)
    err = state.get("error_rate", 1)
    traffic = state.get("traffic_load", 0)

col1, col2, col3 = st.columns(3)

col1.metric("Frontend", frontend)
col2.metric("Backend", backend)
col3.metric("Database", db)

col4, col5 = st.columns(2)
col4.metric("Error Rate", f"{err:.3f}")
col5.metric("Traffic", f"{traffic:.3f}")

# --------------------------------
# LOG
# --------------------------------
st.subheader("Recent Actions")

if st.session_state.log:
    st.table(st.session_state.log[-5:])
else:
    st.write("No actions yet")

# --------------------------------
# STATUS
# --------------------------------
if st.session_state.done:
    if err < 0.05:
        st.success("System Recovered")
    else:
        st.warning("Finished but not fully recovered")
