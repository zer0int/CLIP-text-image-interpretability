import argparse
import subprocess
import os
import sys
import threading
import time

print('\n     1. Running CLIP gradient ascent; this may take some time (~1 min/image with default settings)...\n')
parser = argparse.ArgumentParser(description='Process all images in a directory using the clipga.py script.')
parser.add_argument('--image_dir', type=str, default="IMG_IN", help='The directory containing the images.')
args = parser.parse_args()
if args.image_dir is None:
    raise ValueError("You must provide a path to the image folder using the argument: --image_dir \"path/to/image/folder\"")
def spinning_cursor():
    while not stop_thread:
        for cursor in '|/-\\':
            sys.stdout.write(cursor)
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')

# Get a list of all files in the directory
image_files = os.listdir(args.image_dir)

# Loop over each file
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(args.image_dir, image_file)
    
    # Start the spinning wheel in a separate thread
    stop_thread = False
    t = threading.Thread(target=spinning_cursor)
    t.start()
    
    try:
        # Call the gradient ascent CLIP script with the image path as an argument
        clip_command = ["python", f"clipga.py", "--image_path", image_path]
        result = subprocess.run(clip_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    except KeyboardInterrupt:
        stop_thread = True
        t.join()
        print("\nProcess interrupted by the user.")
        sys.exit(1)
    
    # Stop the spinning wheel thread
    stop_thread = True
    t.join()

    if result.returncode == 0:
        output_filename = f"TOK/tokens_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        print(f"CLIP tokens saved to {output_filename}.")
        # Continue with processing the output tokens
    else:
        print("\nCLIP script encountered an error (below). Continuing with next image (if applicable)...\n\n")
        print("Error details:", result.stderr)
