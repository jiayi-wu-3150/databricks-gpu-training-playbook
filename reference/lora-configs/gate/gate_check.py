import os, subprocess, sys
print("=== python ===", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu[{i}]:", torch.cuda.get_device_name(i))
        print("  capability:", torch.cuda.get_device_capability(i))
        print("  mem GB:", round(torch.cuda.get_device_properties(i).total_memory/1e9, 1))
except Exception as e:
    print("torch import/probe FAILED:", type(e).__name__, e)
print("=== nvidia-smi ===")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout)
except Exception as e:
    print("nvidia-smi FAILED:", e)
for k in ("RANK","WORLD_SIZE","LOCAL_RANK","MASTER_ADDR","NUM_NODES","NODE_RANK"):
    print(f"env {k} =", os.environ.get(k))
try:
    import databricks.air as air
    print("databricks.air OK:", getattr(air, "__version__", "no __version__"))
except Exception as e:
    print("databricks.air import FAILED:", type(e).__name__, e)
