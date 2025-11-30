import os

# Get the absolute path to the results folder
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results", "ba_amici_test_run")

print(f"Checking directory: {results_dir}")

if os.path.exists(results_dir):
    print("Directory exists!")
    files = os.listdir(results_dir)
    if len(files) == 0:
        print("ERROR: The directory is EMPTY. The model did not save.")
    else:
        print("Found files:")
        for f in files:
            print(f" - {f}")
else:
    print("ERROR: Directory does not exist.")