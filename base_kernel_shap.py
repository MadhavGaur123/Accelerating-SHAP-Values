import shap
import numpy as np
import torch
import time

def kernelshap_pixels_gpu(
    model,
    x,            # CIFAR-10 test array: (N, 3072) uint8
    E_hwc,        # (32,32,3) float32 baseline image in [0,1]
    img_idx=21,
    nsamples=9000,
    batch_size=256,          # internal chunking inside f_masks
    device="cuda",
    keep_l1_feature_select=True,  # keep same logic as your current classic call
    show_timing=True,
):
    """
    Classic shap.KernelExplainer on 1024 pixel features (binary masking),
    but with a GPU-friendly f(masks):

      * All image composition in torch on device
      * channels_last + AMP
      * cuDNN autotune on

    Logic kept the same as your reference code:
      - OFF pixels = E_hwc baseline
      - background to KernelExplainer = zeros((1,d))
      - l1_reg=f"num_features(d)" (if keep_l1_feature_select=True)

    Returns:
      phi: (1024,) SHAP values on logit scale
      pred_cls: int (predicted class)
      diag: dict with f(x), f(b), etc.
    """
    assert x.ndim == 2 and x.shape[1] == 32*32*3, "x must be (N, 3072)"
    assert E_hwc.shape == (32, 32, 3), "E_hwc must be (32,32,3) in [0,1]"

    use_cuda = (device == "cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()

    # ---- prep fixed tensors on device (channels_last) ----
    img_chw = x[img_idx].reshape(3, 32, 32).astype(np.float32) / 255.0   # CHW
    base_chw = np.transpose(E_hwc, (2, 0, 1)).astype(np.float32)         # CHW

    img_t  = torch.from_numpy(img_chw).unsqueeze(0).to(device, non_blocking=True)   # (1,3,32,32)
    base_t = torch.from_numpy(base_chw).unsqueeze(0).to(device, non_blocking=True)  # (1,3,32,32)
    img_t  = img_t.contiguous(memory_format=torch.channels_last)
    base_t = base_t.contiguous(memory_format=torch.channels_last)

    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    # ---- get predicted class for the FULL image ----
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
        full_norm = (img_t - mean) / std
        full_norm = full_norm.contiguous(memory_format=torch.channels_last)
        logits_full = model(full_norm)                          # (1,10)
        probs_full  = torch.softmax(logits_full, dim=1)
        pred_cls    = int(torch.argmax(probs_full, dim=1).item())

    d = 32 * 32

    # ---- GPU-friendly f(masks) ----
    def f_masks(masks: np.ndarray):
        """
        masks: (B,1024) float32 in {0,1}
        Return logits[:, pred_cls] for composites built as:
            composite = mask*img + (1-mask)*baseline
        """
        masks = np.asarray(masks, dtype=np.float32)
        B = masks.shape[0]
        out = np.empty((B,), dtype=np.float32)

        # process in mini-batches to control memory
        for s in range(0, B, batch_size):
            e = min(s + batch_size, B)
            mb_np = masks[s:e].reshape(-1, 1, 32, 32)          # (b,1,32,32)
            mb = torch.from_numpy(mb_np).to(device, non_blocking=True)
            mb = mb.contiguous(memory_format=torch.channels_last)

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_cuda):
                imgs = mb * img_t + (1.0 - mb) * base_t        # (b,3,32,32) broadcast
                imgs = (imgs - mean) / std
                imgs = imgs.contiguous(memory_format=torch.channels_last)
                logits = model(imgs)                            # (b,10)
                z = logits[:, pred_cls].detach().float().cpu().numpy()
            out[s:e] = z
        return out

    # ---- KernelExplainer wiring identical to your original logic ----
    background = np.zeros((1, d), dtype=np.float32)  # empty coalition (all OFF)
    expl = shap.KernelExplainer(f_masks, background, link="identity")

    x_full = np.ones((1, d), dtype=np.float32)       # explain full coalition (all ON)

    if keep_l1_feature_select:
        l1_reg = f"num_features({d})"                # SAME as your current code
    else:
        l1_reg = 0.0                                 # faster, but changes the solver path

    phi = expl.shap_values(x_full, nsamples=nsamples, l1_reg=l1_reg)
    phi = np.array(phi, dtype=np.float32).reshape(d)

    # ---- diagnostics (same semantics) ----
    fx = float(f_masks(x_full)[0])
    fb = float(f_masks(np.zeros((1, d), dtype=np.float32))[0])

    if show_timing:
        if use_cuda:
            torch.cuda.synchronize()
        print(f"[Timing] KernelSHAP total: {time.perf_counter() - t0:.2f}s")

    print("\n[KernelSHAP (classic) — LOGIT]")
    print(f"Predicted class: {pred_cls}")
    print(f"f(x) (full logit)    : {fx:.6f}")
    print(f"f(b) (baseline logit): {fb:.6f}")
    print(f"f(x) - f(b)          : {fx - fb:.6f}")
    print(f"Sum SHAP (pixels)    : {phi.sum():.6f}")

    return phi, pred_cls, {"f(x)": fx, "f(b)": fb, "fx_minus_fb": fx - fb, "sum_phi": float(phi.sum())}
_is_cuda = torch.cuda.is_available()
_start = time.perf_counter()

phi_1024, pred_cls, check = kernelshap_pixels_gpu(
    model=model,
    x=x,
    E_hwc=E_hwc,
    img_idx=21,
    nsamples=3000,
    batch_size=256,                 # internal chunking for masks
    device="cuda" if _is_cuda else "cpu",
    keep_l1_feature_select=True,    # keep EXACT classic behavior
    show_timing=True
)

np.set_printoptions(suppress=True, linewidth=120, precision=6)
print("\nKernelSHAP SHAP (1024) LOGIT:")
print(phi_1024)
print("Length:", len(phi_1024))
print("Sum:", phi_1024.sum())

if _is_cuda:
    torch.cuda.synchronize()
_total = time.perf_counter() - _start
print(f"\n[Timing] Total script time: {_total:.2f} s")
