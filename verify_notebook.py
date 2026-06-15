import json
import sys

# Load the notebook
with open('churn_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract code cells
code_blocks = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Reconstruct the code string
        code_lines = cell['source']
        code_blocks.append("".join(code_lines))

# Concatenate all code cells
full_script = "import matplotlib\nmatplotlib.use('Agg')\n\n" + "\n# === CELL BOUNDARY ===\n".join(code_blocks)

# Write to a temporary file
verify_file = 'verify_run.py'
with open(verify_file, 'w', encoding='utf-8') as f:
    f.write(full_script)

print(f"Generated python verification script: {verify_file}")

# Try to run the generated script
import subprocess

try:
    print("Executing verification script...")
    result = subprocess.run([sys.executable, verify_file], capture_output=True, text=True, check=True)
    print("Verification script executed successfully!")
    print("\n--- STDOUT ---")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("ERROR executing verification script!")
    print("\n--- STDOUT ---")
    print(e.stdout)
    print("\n--- STDERR ---")
    print(e.stderr)
    sys.exit(1)
