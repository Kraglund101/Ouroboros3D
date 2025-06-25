import os
import zipfile

# Define paths
zip_path = "to_server_delete_after.zip"

print("Checking what needs to be extracted...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    files_to_extract = []
    files_skipped = 0
    
    for file_info in zip_ref.infolist():
        file_path = file_info.filename
        
        # Check if this file already exists
        if os.path.exists(file_path):
            files_skipped += 1
        else:
            files_to_extract.append(file_info)
    
    print(f"Files already exist: {files_skipped}")
    print(f"Files to extract: {len(files_to_extract)}")
    
    # Extract only the missing files
    if files_to_extract:
        print("Extracting missing files...")
        for file_info in files_to_extract:
            zip_ref.extract(file_info, ".")
        print("Extraction complete!")
    else:
        print("All files already exist - nothing to extract")

# Rest of the script - generate txt files
if os.path.exists("data/train") and os.path.exists("data/val"):
    print("✅ Found both data/train and data/val directories")
    
    train_dir = "data/train"
    val_dir = "data/val"
    
    train_file = "data/render-o3d-train.txt"
    val_file = "data/render-o3d-val.txt"
    
    # Get folder names
    print("Scanning train directory...")
    train_ids = sorted([
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name)) and not name.startswith(".")
    ])
    
    print("Scanning val directory...")
    val_ids = sorted([
        name for name in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, name)) and not name.startswith(".")
    ])
    
    # Write files
    with open(train_file, "w") as f:
        for scene in train_ids:
            f.write(f"{scene}\n")
    
    with open(val_file, "w") as f:
        for scene in val_ids:
            f.write(f"{scene}\n")
    
    print(f"✅ Created {train_file} with {len(train_ids)} scenes.")
    print(f"✅ Created {val_file} with {len(val_ids)} scenes.")
else:
    print("❌ Missing train or val directories")