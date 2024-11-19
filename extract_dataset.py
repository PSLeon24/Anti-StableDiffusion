import os
import shutil

def move_jpg_files():
    current_dir = os.getcwd()
    
    for root, dirs, files in os.walk(current_dir, topdown=False):
        for file in files:
            if file.lower().endswith('.jpg'):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(current_dir, file)
                
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_name = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(current_dir, new_name)
                        counter += 1

                shutil.move(source_path, dest_path)

        if root != current_dir:
            try:
                os.rmdir(root)
            except OSError:
                print(f"This folder cannot be deleted: {root}")

if __name__ == "__main__":
    move_jpg_files()
    print("Done")

