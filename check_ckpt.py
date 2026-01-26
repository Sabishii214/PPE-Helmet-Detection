import torch
from pathlib import Path

def check_checkpoint(path):
    print(f"Checking checkpoint: {path}")
    if not Path(path).exists():
        print("File does not exist.")
        return False
    
    try:
        torch.load(path, map_location='cpu', weights_only=False)
        print("SUCCESS: Checkpoint is valid.")
        return True
    except RuntimeError as e:
        if "failed finding central directory" in str(e) or "PytorchStreamReader" in str(e):
            print(f"CORRUPTION DETECTED: The file is truncated or corrupted.")
            print(f"Error detail: {e}")
            return False
        else:
            # Other errors (like the safely load one) might just be custom classes
            print(f"Note: Encountered a potential load issue, but not necessarily zip corruption: {e}")
            return True # Treat as potentially valid but needing custom loader
    except Exception as e:
        print(f"Error checking checkpoint: {e}")
        return False

checkpoint_path = 'output/train/weights/last.pt'
if not check_checkpoint(checkpoint_path):
    print("\nAction Required:")
    print(f"The file {checkpoint_path} is corrupted and cannot be used to resume training.")
    print("You should remove it and start the training again.")
    print(f"Run: rm {checkpoint_path}")
else:
    print("\nThe checkpoint appears to be physically valid (not corrupted at the ZIP level).")
