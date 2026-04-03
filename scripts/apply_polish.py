#!/usr/bin/env python3

file = "app.py"

with open(file, "r") as f:
    code = f.read()

# 1. Add status badge function at the top
status_block = '''def render_status(val):
    """Convert numeric status to emoji badge."""
    if int(val) == 2:
        return "🟢 Healthy"
    elif int(val) == 1:
        return "🟡 Degraded"
    else:
        return "🔴 Down"

'''

if "def render_status" not in code:
    # Insert after imports
    import_end = code.find("st.set_page_config")
    code = code[:import_end] + status_block + code[import_end:]

# 2. Replace status displays with render_status calls
code = code.replace(
    'int(state[0])',
    'render_status(state[0])'
)
code = code.replace(
    'int(state[1])',
    'render_status(state[1])'
)
code = code.replace(
    'int(state[2])',
    'render_status(state[2])'
)

# 3. Add decision explanation panel before "Recent Actions"
explain_block = '''st.subheader("AI Decision Insight")

last_action = st.session_state.log[-1]["action"] if st.session_state.log else None

ACTION_REASON = {
    0: "System stable → monitoring",
    1: "Frontend issue detected → restarting frontend",
    2: "Backend instability → restarting backend",
    3: "Database failure → highest priority restart",
    4: "High load → throttling traffic",
    5: "System imbalance → rebalancing traffic"
}

if last_action is not None:
    st.info(ACTION_REASON.get(last_action, "Monitoring system"))

'''

if "AI Decision Insight" not in code:
    code = code.replace(
        'st.subheader("Recent Actions")',
        explain_block + 'st.subheader("Recent Actions")'
    )

# 4. Add critical warning after "System State"
warning_block = '''
if any(int(state[i]) == 0 for i in [0,1,2]):
    st.error("⚠ Critical failure detected in system!")
'''

if "Critical failure detected" not in code:
    code = code.replace(
        'st.subheader("System State")',
        'st.subheader("System State")' + warning_block
    )

# 5. Add performance summary at the end
summary_block = '''
st.subheader("Performance Summary")

state = st.session_state.state
steps = st.session_state.step
try:
    error = state[6]
except:
    error = state.get("error_rate", 1.0) if isinstance(state, dict) else 1.0

if error < 0.01:
    st.success(f"✅ Recovered in {steps} steps with low error rate {error:.3f}")
elif error < 0.05:
    st.warning(f"⚠️ Partially stable after {steps} steps (error: {error:.3f})")
else:
    st.error(f"❌ System not stable (error: {error:.3f})")
'''

if "Performance Summary" not in code:
    code = code + "\n\n" + summary_block

# 6. Limit log to 10 entries (clean UI)
code = code.replace(
    "st.session_state.log.append({",
    "st.session_state.log = st.session_state.log[-10:]\n    st.session_state.log.append({"
)

# Save
with open(file, "w") as f:
    f.write(code)

print("✅ FINAL DEMO POLISH APPLIED SUCCESSFULLY")
