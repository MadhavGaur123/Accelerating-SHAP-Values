import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap

# ---- SLIC (superpixels) ----
from skimage.segmentation import slic
from skimage.util import img_as_float

# ------------------ CONFIG ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"/kaggle/input/model-model/cifar10-resnet18best.pth"
data_path  = r"/kaggle/input/cifar-10-batches-py"
test_path  = os.path.join(data_path, "test_batch")
background_path = os.path.join(data_path,"data_batch_1")
background_path_two = os.path.join(data_path,"data_batch_2")
background_path_three = os.path.join(data_path,"data_batch_3")
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

np.seterr(all='warn')
torch.backends.cudnn.benchmark = True

# ------------------ MODEL ------------------
def conv_block(in_channels, out_channels, activation=False, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels)
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Sequential(
            conv_block(64, 64, activation=True),
            conv_block(64, 64)
        )
        self.res2 = nn.Sequential(
            conv_block(64, 64, activation=True),
            conv_block(64, 64)
        )
        self.downsample1 = nn.Sequential(conv_block(64, 128, pool=True))
        self.res3 = nn.Sequential(
            conv_block(64, 128, activation=True, pool=True),
            conv_block(128, 128)
        )
        self.res4 = nn.Sequential(
            conv_block(128, 128, activation=True),
            conv_block(128, 128, activation=True)
        )
        self.res5 = nn.Sequential(
            conv_block(128, 256, activation=True, pool=True),
            conv_block(256, 256)
        )
        self.downsample2 = nn.Sequential(
            conv_block(128, 256, pool=True, activation=True)
        )
        self.res6 = nn.Sequential(
            conv_block(256, 256, activation=True),
            conv_block(256, 256, activation=True)
        )
        self.res7 = nn.Sequential(
            conv_block(256, 512, activation=True, pool=True),
            conv_block(512, 512, activation=True)
        )
        self.downsample3 = nn.Sequential(
            conv_block(256, 512, activation=True, pool=True)
        )
        self.res8 = nn.Sequential(
            conv_block(512, 512, activation=True),
            conv_block(512, 512, activation=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + self.downsample1(out)
        out = self.res4(out) + out
        out = self.res5(out) + self.downsample2(out)
        out = self.res6(out) + out
        out = self.downsample3(out) + self.res7(out)
        out = self.res8(out) + out
        out = self.classifier(out)
        return out

# ------------------ LOAD MODEL ------------------
model = ResNet18(in_channels=3, num_classes=10).to(device)
state = torch.load(model_path, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()

# ------------------ LOAD CIFAR-10 & BASELINE ------------------
with open(test_path, "rb") as f:
    batch = pickle.load(f, encoding="latin1")
x = batch["data"]
y = batch["labels"]
fnames = batch["filenames"]

with open(background_path, "rb") as f:
    background = pickle.load(f, encoding="latin1")
with open(background_path_two, "rb") as f:
    background_two = pickle.load(f, encoding="latin1")
with open(background_path_three, "rb") as f:
    background_three = pickle.load(f, encoding="latin1")

background_image_data = np.concatenate(
    (background["data"], background_two["data"], background_three["data"]),
    axis=0
)
bg = background_image_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
E_chw = bg.mean(axis=0)                        # (3,32,32)
E_hwc = np.transpose(E_chw, (1,2,0)).copy()    # (32,32,3)

# ------------------ UTILS ------------------
def find_last_conv(module: nn.Module):
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d layer found.")

def predict_class(img_idx, device="cuda"):
    img = x[img_idx].reshape(3,32,32).astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1,3,1,1)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1,3,1,1)
    t = torch.from_numpy(img).unsqueeze(0).to(device)
    t = (t - torch.from_numpy(mean).to(device)) / torch.from_numpy(std).to(device)
    with torch.inference_mode():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1)
        pred   = int(torch.argmax(probs, dim=1).item())
        conf   = float(probs[0, pred].item())
    return pred, conf

def gradcam_heatmap(img_idx, show=False, device="cuda"):
    """
    Returns:
      pred_cls (int), grayscale_cam (32,32) float in [0,1-ish]
    """
    img = x[img_idx].reshape(3,32,32).astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3,1,1)
    img_norm = (img - mean) / std
    img_t = torch.from_numpy(img_norm).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(img_t)
        probs  = torch.softmax(logits, dim=1)
        pred   = int(torch.argmax(probs, dim=1).item())

    target_layer = find_last_conv(model)
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(
        input_tensor=img_t,
        targets=targets,
        eigen_smooth=True,
        aug_smooth=True
    )[0]  # (32,32)

    if show:
        overlay = show_cam_on_image(
            np.clip(np.transpose(img, (1,2,0)), 0, 1),
            grayscale_cam,
            use_rgb=True
        )
        plt.figure(figsize=(7.5,2.5))
        plt.subplot(1,3,1); plt.title("Input"); plt.imshow(np.transpose(img,(1,2,0))); plt.axis('off')
        plt.subplot(1,3,2); plt.title("Grad-CAM++"); plt.imshow(grayscale_cam, cmap='jet'); plt.axis('off')
        plt.subplot(1,3,3); plt.title("Overlay"); plt.imshow(overlay); plt.axis('off')
        plt.tight_layout(); plt.show()

    return pred, grayscale_cam

def predict_logits_of_class(imgs_hwc, target_cls, device="cuda", batch_size=256):
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1,1,1,3)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1,1,1,3)
    outs = []
    with torch.inference_mode():
        for s in range(0, len(imgs_hwc), batch_size):
            chunk = np.stack(imgs_hwc[s:s+batch_size], axis=0).astype(np.float32)
            chunk = (chunk - mean) / std
            chunk = np.transpose(chunk, (0,3,1,2))  # BCHW
            t = torch.from_numpy(chunk).to(device)
            logits = model(t)
            outs.append(logits[:, target_cls].detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)

# ============================================================
# KernelSHAP with SLIC superpixels, keeping your logic identical
# ============================================================
def kernelshap_topk_slic(
    img_idx,
    n_segments=100,          # SLIC target segments (actual M may differ slightly)
    compactness=10.0,
    sigma=0.0,
    k=60,                    # IMPORTANT: k = number of superpixels selected (not pixels)
    nsamples=3000,
    device="cuda",
    batch_size=256,
    seed=2025,
    show_cam=False,
    show_segments=False,
    cam_frac=0.6
):
    """
    SAME LOGIC as your pixel code, but "features" are SLIC superpixels.

    Selection:
      - cam_frac*k superpixels with highest mean Grad-CAM heat
      - (1-cam_frac)*k random superpixels from the remaining

    Fixed-context game:
      - non-selected superpixels are ALWAYS original
      - selected superpixels toggle baseline(E_hwc) vs original based on coalition mask

    Output:
      - phi_full_shap (1024,) mapped back to pixels (sum-preserving distribution within each superpixel)
        (non-selected superpixels -> 0 contribution, like your pixel scatter)
    """
    rng = np.random.default_rng(seed)

    # load image
    img_chw = x[img_idx].reshape(3,32,32).astype(np.float32) / 255.0
    img_hwc = np.transpose(img_chw, (1,2,0)).copy()

    # predicted class
    pred_cls, conf = predict_class(img_idx, device=device)
    print(f"Image {img_idx}: pred={pred_cls} ({CIFAR10_CLASSES[pred_cls]}), conf={conf:.2%}")

    # Grad-CAM heatmap for PRIOR
    _, heat = gradcam_heatmap(img_idx, show=show_cam, device=device)  # (32,32)

    # SLIC superpixels
    segments = slic(
        img_as_float(img_hwc),
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1
    )
    M = int(segments.max() + 1)
    print(f"[SLIC] requested n_segments={n_segments} -> actual M={M}")

    if show_segments:
        plt.figure(figsize=(4,4))
        plt.title("SLIC segments (labels)")
        plt.imshow(segments)
        plt.axis("off")
        plt.show()

    # Score each superpixel by mean CAM heat inside it
    seg_scores = np.zeros((M,), dtype=np.float32)
    for s in range(M):
        mask = (segments == s)
        if mask.any():
            seg_scores[s] = float(heat[mask].mean())
        else:
            seg_scores[s] = -1e9

    seg_ranked = np.argsort(-seg_scores).astype(np.int64)  # highest heat first

    # -------- mix cam-selected + random superpixels (SAME as your pixel logic) --------
    k = min(int(k), M)
    cam_frac = max(0.0, min(1.0, float(cam_frac)))

    k_cam  = int(round(k * cam_frac))
    k_cam  = min(k_cam, k)
    k_rand = k - k_cam

    cam_keep = seg_ranked[:k_cam].astype(np.int64)

    if k_rand > 0:
        all_seg = np.arange(M, dtype=np.int64)
        remaining = np.setdiff1d(all_seg, cam_keep, assume_unique=False)
        k_rand = min(k_rand, len(remaining))
        rand_keep = rng.choice(remaining, size=k_rand, replace=False).astype(np.int64)
        selected_seg = np.concatenate([cam_keep, rand_keep], axis=0)
    else:
        selected_seg = cam_keep

    k_eff = len(selected_seg)
    print(f"\n[Top-k selection on SUPERPIXELS] total k={k_eff} | cam_frac={cam_frac:.2f} => cam={k_cam}, random={k_rand}")
    # -------------------------------------------------------------------------------

    # Build masked image from coalition mask over selected superpixels
    def img_from_mask_k(mask_k):
        # fixed-context: start as ORIGINAL everywhere (non-selected superpixels stay original)
        out_img = img_hwc.copy()
        # for selected superpixels: if mask=0 -> baseline, if mask=1 -> keep original
        for j, seg_id in enumerate(selected_seg):
            if mask_k[j] < 0.5:
                out_img[segments == seg_id] = E_hwc[segments == seg_id]
        return out_img

    def f_masks_k(mask_batch_k):
        imgs = [img_from_mask_k(m.astype(np.float32)) for m in mask_batch_k]
        return predict_logits_of_class(imgs, pred_cls, device=device, batch_size=batch_size)

    # background: selected OFF (baseline), non-selected implicitly original (because we start from img_hwc)
    background_k = np.zeros((1, k_eff), dtype=np.float32)
    expl = shap.KernelExplainer(f_masks_k, background_k, link="identity")

    # full coalition: selected ON -> full real image
    x_full_k = np.ones((1, k_eff), dtype=np.float32)

    phi_k_shap = expl.shap_values(
        x_full_k,
        nsamples=int(nsamples),
        l1_reg=f"num_features({k_eff})"
    )
    phi_k_shap = np.array(phi_k_shap, dtype=np.float32).reshape(k_eff)

    # Map superpixel SHAP to pixel SHAP (1024,)
    # IMPORTANT: to avoid blowing up sums, distribute each superpixel value equally across its pixels.
    phi_map = np.zeros((32,32), dtype=np.float32)  # per-pixel map
    for j, seg_id in enumerate(selected_seg):
        mask = (segments == seg_id)
        cnt = int(mask.sum())
        if cnt > 0:
            phi_map[mask] = phi_k_shap[j] / float(cnt)

    phi_full_from_shap = phi_map.reshape(-1).astype(np.float32)

    # Diagnostics consistent with your code
    fx = float(f_masks_k([np.ones((k_eff,), dtype=np.float32)])[0])   # real img
    fb = float(f_masks_k([np.zeros((k_eff,), dtype=np.float32)])[0])  # selected baseline, others original
    print("\n[Fixed-context KernelSHAP on SLIC superpixels]")
    print(f"f(x) (logit, real img): {fx:.6f} | f(b) (logit, selected baseline): {fb:.6f} | f(x)-f(b): {fx - fb:.6f}")
    print(f"Sum φ_k (shap): {phi_k_shap.sum():.6f} | Sum φ_full (pixel-mapped): {phi_full_from_shap.sum():.6f}")

    diag = {
        "pred_cls": pred_cls,
        "f_x": fx,
        "f_b": fb,
        "fx_minus_fb": fx - fb,
        "sum_phi_k_shap": float(phi_k_shap.sum()),
        "k_eff": int(k_eff),
        "nsamples": int(nsamples),
        "cam_frac": float(cam_frac),
        "k_cam": int(k_cam),
        "k_rand": int(k_rand),
        "M_superpixels": int(M),
        "n_segments_requested": int(n_segments),
        "compactness": float(compactness),
        "sigma": float(sigma),
    }

    return {
        "phi_k_shap": phi_k_shap,                 # per selected superpixel (k,)
        "phi_full_shap": phi_full_from_shap,      # per pixel (1024,)
        "selected_superpixels": selected_seg,     # ids of selected superpixels
        "segments": segments,                     # (32,32) labels
        "diag": diag
    }

# ------------------ RUN ------------------
if __name__ == "__main__":
    t0 = time.perf_counter()

    out = kernelshap_topk_slic(
        img_idx=21,
        # SLIC controls
        n_segments=100,
        compactness=10.0,
        sigma=0.0,

        # your selection logic (NOW in superpixel-space)
        k=60,               # e.g., 60 superpixels out of ~100
        nsamples=1000,
        device=device,
        batch_size=256,
        seed=2025,
        show_cam=False,
        show_segments=False,
        cam_frac=0.6
    )

    phi_shap = out["phi_full_shap"]
    print("\n[Done]")
    print("φ_full_shap length:", len(phi_shap), "sum:", float(phi_shap.sum()))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Total time: {time.perf_counter()-t0:.2f}s")

