import os
import shutil


def create_or_reset_dir(path):
    # Create save_dir
    if not os.path.exists(path):
        os.makedirs(path)
    # Delete old contents of path
    for files in os.listdir(path):
        path = os.path.join(path, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
