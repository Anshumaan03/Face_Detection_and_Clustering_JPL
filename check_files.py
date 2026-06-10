import os

target_dir = "/Users/anshumaansinghrathore/Desktop/Face Clustering/data/raw"

print(f"1. Does the directory exist? {os.path.exists(target_dir)}")

if os.path.exists(target_dir):
    contents = os.listdir(target_dir)
    print(f"2. Number of items inside this folder: {len(contents)}")
    print(f"3. Raw list of items: {contents}")
    
    # Check what's inside the very first item if it's a directory
    for item in contents:
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            print(f"   👉 Found subfolder: '{item}', containing: {os.listdir(item_path)}")