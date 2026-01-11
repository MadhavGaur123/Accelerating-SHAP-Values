from __future__ import annotations
import importlib
import sys
import platform

REQUIREMENTS = [
    ("numpy",              "numpy",              "1.20.0", "NumPy"),
    ("torch",              "torch",              "1.13.0", "PyTorch"),
    ("matplotlib",         "matplotlib",         "3.4.0",  "Matplotlib"),
    ("shap",               "shap",               "0.41.0", "SHAP"),
    ("pytorch_grad_cam",   "pytorch-grad-cam",   "1.4.8",  "pytorch-grad-cam"),
    ("skimage",            "scikit-image",       "0.19.0", "scikit-image"),
]

def _parse_version(v: str):
    import re
    nums = re.findall(r"\d+", v)
    nums = [int(x) for x in nums[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)

def _meets_min(actual: str, minimum: str | None) -> bool:
    if not minimum:
        return True
    try:
        return _parse_version(actual) >= _parse_version(minimum)
    except Exception:
        return True

def _get_version(mod):
    v = getattr(mod, "__version__", None)
    if v is None and hasattr(mod, "version"):
        v = getattr(mod.version, "__version__", None)
    return v or "unknown"

def _nice(s: str, width: int) -> str:
    s = str(s)
    return (s if len(s) <= width else s[:width-1] + "…").ljust(width)

def check_packages():
    print("="*72)
    print("Environment & Dependencies Check")
    print("="*72)
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    print(f"OS          : {platform.system()} {platform.release()} | {platform.platform()}")
    print("-"*72)

    rows = []
    missing = []
    outdated = []

    for import_name, pip_name, min_ver, friendly in REQUIREMENTS:
        try:
            mod = importlib.import_module(import_name)
            ver = _get_version(mod)
            ok = _meets_min(ver, min_ver)
            rows.append((friendly, ver, min_ver or "-", "OK" if ok else "Upgrade"))
            if not ok:
                outdated.append((pip_name, min_ver))
        except Exception:
            rows.append((friendly, "—", min_ver or "-", "Missing"))
            missing.append(pip_name)

    colw = (22, 14, 12, 10)
    headers = ("Package", "Installed", "Minimum", "Status")
    print(_nice(headers[0], colw[0]), _nice(headers[1], colw[1]), _nice(headers[2], colw[2]), _nice(headers[3], colw[3]))
    print("-"*sum(colw))
    for r in rows:
        print(_nice(r[0], colw[0]), _nice(r[1], colw[1]), _nice(r[2], colw[2]), _nice(r[3], colw[3]))
    print("-"*sum(colw))
    try:
        import torch
        print("PyTorch CUDA diagnostics")
        print(f"  torch.version        : {torch.__version__}")
        print(f"  CUDA available       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device count    : {torch.cuda.device_count()}")
            print(f"  Current device       : {torch.cuda.current_device()}")
            print(f"  Device name          : {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"  cuDNN enabled        : {torch.backends.cudnn.enabled}")
            print(f"  cuDNN version        : {getattr(torch.backends.cudnn, 'version', lambda: 'unknown')()}")
        else:
            print("  (If you expected GPU, ensure the runtime has CUDA and correct torch build)")
    except Exception:
        print("PyTorch not importable; skipping CUDA diagnostics.")

    print("-"*72)
    if missing or outdated:
        print("Suggested pip commands:")
        combo = []
        for name in missing:
            combo.append(name)
        for name, minv in outdated:
            combo.append(f"{name}>={minv}")
        if combo:
            print("  pip install -U " + " ".join(combo))
        for name in missing:
            print(f"  pip install -U {name}")
        for name, minv in outdated:
            print(f"  pip install -U '{name}>={minv}'")
    else:
        print("All good: required packages are present and meet minimum versions.")

    print("="*72)

if __name__ == "__main__":
    check_packages()
