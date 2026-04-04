import re

file_path = "inference.py"

with open(file_path, "r") as f:
    code = f.read()

# Remove unwanted prints
code = re.sub(r'print\("===.*?===.*?"\)', '', code)
code = re.sub(r'print\(f?"===.*?===.*?"\)', '', code)
code = re.sub(r'print\(.*SUMMARY.*\)', '', code)

# Replace step print pattern
code = re.sub(
    r'print\(f?"Step\s*\{?(\w+)\}?\s*\|\s*Action:\s*\{?(\w+)\}?\s*\|\s*Score:\s*\{?(\w+)\}?.*?"\)',
    r'print(f"[STEP] step={\1} action={\2} score={\3}")',
    code
)

# Add START before task loops (basic safe insert)
code = code.replace(
    "for task_name",
    'print(f"[START] task={task_name}")\nfor task_name'
)

# Replace final score prints
code = re.sub(
    r'print\(f?"Final Score:\s*\{?(\w+)\}?.*?"\)',
    r'print(f"[END] final_score={\1} steps={step}")',
    code
)

with open(file_path, "w") as f:
    f.write(code)

print("inference.py format updated")
