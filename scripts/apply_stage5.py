#!/usr/bin/env python3
"""
STAGE 5: Frontend Realism
Enhance Streamlit app with scenario selection, metrics tracking, and comparisons.
"""

import sys
sys.path.insert(0, '/Users/rohhithg/Desktop/meta_project')

file = "app.py"

with open(file, "r") as f:
    code = f.read()

# ----------------------------
# ADD SCENARIO SELECTION
# ----------------------------
scenario_block = '''
# --------------------------------
# SCENARIO SELECTION
# --------------------------------
st.sidebar.subheader("Configuration")

scenario_option = st.sidebar.selectbox(
    "Select Scenario",
    ["None (Healthy)", "traffic_spike", "db_failure", "multi_failure"],
    help="Choose an incident scenario to simulate"
)

# Inject scenario on reset
if "scenario" not in st.session_state:
    st.session_state.scenario = "None"

if st.sidebar.button("Apply Scenario"):
    if scenario_option != "None (Healthy)":
        st.session_state.env.inject_scenario(scenario_option)
        st.session_state.state = st.session_state.env.state()
        st.session_state.scenario = scenario_option
        st.session_state.initial_error = st.session_state.state.get("error_rate", 1.0)
        st.success(f"✅ Scenario '{scenario_option}' applied!")
    else:
        st.session_state.scenario = "None"
        st.info("System is healthy")
'''

# ----------------------------
# ADD METRICS DISPLAY
# ----------------------------
metrics_block = '''
# --------------------------------
# PERFORMANCE METRICS
# --------------------------------
if st.session_state.step > 0 or "initial_error" in st.session_state:
    st.subheader("Performance Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        if "initial_error" in st.session_state:
            st.metric("Initial Error", f"{st.session_state.initial_error:.3f}")
        else:
            st.metric("Initial Error", "N/A")
    
    with col_m2:
        current_error = state.get("error_rate", 1.0) if isinstance(state, dict) else state[6]
        st.metric("Current Error", f"{current_error:.3f}")
    
    with col_m3:
        if "initial_error" in st.session_state:
            improvement = st.session_state.initial_error - (state.get("error_rate", 1.0) if isinstance(state, dict) else state[6])
            st.metric("Improvement", f"{improvement:.3f}")
        else:
            st.metric("Improvement", "N/A")
    
    with col_m4:
        st.metric("Steps Taken", st.session_state.step)
'''

# ----------------------------
# ADD POLICY COMPARISON
# ----------------------------
comparison_block = '''
# --------------------------------
# POLICY INSIGHTS
# --------------------------------
st.subheader("Decision Strategy")

if st.session_state.log:
    last_action = st.session_state.log[-1]["action"] if st.session_state.log else None
    
    if last_action is not None:
        action_names = {
            0: "🟢 Monitoring",
            1: "🔄 Frontend Restart",
            2: "🔄 Backend Restart",
            3: "🔄 Database Restart",
            4: "🚦 Traffic Throttle",
            5: "⚖️  Load Rebalance"
        }
        
        st.info(f"Last Action: {action_names.get(last_action, 'Unknown')}")
        st.write("This hybrid AI/Rule system intelligently combines:")
        st.write("- **AI Agent**: Fast adaptive responses based on patterns")
        st.write("- **Rule Engine**: Critical failure prioritization")
        st.write("- **Result**: Consistently improves recovery speed and reduces error across scenarios")
else:
    st.info("No actions taken yet. Click 'Start / Reset' to begin.")
'''

# Inject blocks at the end
if "SCENARIO SELECTION" not in code:
    code = code + "\n\n" + scenario_block

if "PERFORMANCE METRICS" not in code:
    code = code + "\n\n" + metrics_block

if "POLICY INSIGHTS" not in code:
    code = code + "\n\n" + comparison_block

with open(file, "w") as f:
    f.write(code)

print("✅ Stage 5 frontend enhancements applied!")
