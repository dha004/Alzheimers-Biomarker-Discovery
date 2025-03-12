import subprocess
import os

# Define the path to the scripts directory
scripts_dir = "scripts"

# List of scripts to execute in order
scripts = [
    "preprocess.py",
    "biomarker_identification.py",
    "train_model.py",
    "analyze_results.py"
]

print("\n Starting the pipeline...\n")

# Loop through each script and execute it
for script in scripts:
    script_path = os.path.join(scripts_dir, script)

    print(f"\n Running {script_path}...\n")

    # Run the script and wait for it to complete
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    # Print script output
    print(result.stdout)

    # Check for errors
    if result.returncode != 0:
        print(f" Error in {script}:\n{result.stderr}")
        break  # Stop execution if a script fails

print("\n Pipeline execution complete.")