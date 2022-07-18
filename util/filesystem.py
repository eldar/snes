import os

def mkdir_shared(path, exist_ok=True):
    if not path.exists():
        path.mkdir()
        os.chmod(path, 0o777)
