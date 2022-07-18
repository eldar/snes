from pathlib import Path

def delete_old_checkpoints(ckpt_dir, debug_print=False):
    ckpt_dir = Path(ckpt_dir)
    checkpoints = [c for c in ckpt_dir.iterdir() if c.is_file()]
    checkpoints = [c for c in checkpoints if c.suffix == ".pth"]
    checkpoints.sort()
    checkpoints.pop() # leave the last checkpoint
    for ckpt in checkpoints:
        if debug_print:
            print(f"Deleting {ckpt}")
        ckpt.unlink()
