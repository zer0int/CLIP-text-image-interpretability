import os
import subprocess

# 1: Check if the folders "VIS", "TOK", and "IMG_IN" exist. If not, create them in the current directory
directories = ["VIS", "TOK", "IMG_IN"]
for dir_name in directories:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 2: Check if the "IMG_IN" folder is empty
while not os.listdir("IMG_IN"):
    input("Please put some images into the IMG_IN folder. Press enter to continue...")

# 3: Execute the scripts in order
scripts_to_run = ["runclipga.py", "runexplain.py", "runmakenames.py", "runmakemosaic.py"]
for script in scripts_to_run:
    process = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
    # Stream the stdout in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    # Wait for the process to finish and get the returncode
    return_code = process.wait()
    
    if return_code != 0:
        error_output = process.stderr.read()
        print(f"Error occurred while running {script}. Error details:", error_output)
        break

# 4: Print the final message
print("\n\nDONE. Check the 'VIS' folder for the results!")