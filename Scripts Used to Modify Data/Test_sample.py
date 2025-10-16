import os
import random
import shutil

source_dir = "C:/Matura/data"
target_dir = "C:/Matura/testing"
num_files_to_move = 100

# List all files in source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Randomly select files to move
files_to_move = random.sample(files, num_files_to_move)

# Move selected files to target directory
for f in files_to_move:
    shutil.move(os.path.join(source_dir, f), os.path.join(target_dir, f))

# Helper function to rename files sequentially in a directory
def renumber_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # Sort to preserve order
    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{i}{ext}"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

# Renumber files in both directories
renumber_files(source_dir)
renumber_files(target_dir)

print(f"Moved {num_files_to_move} files from {source_dir} to {target_dir}")
print(f"Renamed files in {source_dir} to 1-{len(os.listdir(source_dir))}")
print(f"Renamed files in {target_dir} to 1-{len(os.listdir(target_dir))}")
