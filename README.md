# Prior-Guided KernelSHAP for Images (CIFAR-10, ResNet-18)

This repo explores a **faster approximation of KernelSHAP** for image models by using a **prior heatmap** (Grad-CAM++ / DeepLIFT) to reduce the feature space before running KernelSHAP.

✅ **Main idea:** KernelSHAP over all pixels is expensive (1024 features → huge coalition space).  
🚀 We speed it up by **first using a prior** to pick **top-k “important” features** (pixels or superpixels), and then running **KernelSHAP only on those k features** while keeping everything else fixed to the original image.

---

## What “normal” KernelSHAP does (for an image)

A 32×32 image has **1024 pixel locations**. In classic KernelSHAP we treat each pixel location as a feature:

- **ON (1):** keep the original pixel  
- **OFF (0):** replace it with a baseline value (we use the mean CIFAR-10 image `E_hwc`)

KernelSHAP then:

1. **Samples coalitions** (binary masks of which pixels are ON vs OFF).
2. Builds masked images:  
   `composite = mask * image + (1 - mask) * baseline`
3. Runs the model on each masked image to get a **score** (here: **logit of predicted class**).
4. Solves a **Weighted Least Squares (WLS)** regression using Shapley kernel weights to estimate **Shapley values** per pixel.
5. Enforces additivity (approximately):  
   `sum(phi) ≈ f(full_image) - f(baseline)`

**Accurate but slow** because the feature space is large (1024 features).

---

## Our faster method: prior-guided Top-k KernelSHAP

We keep the KernelSHAP idea, but **we do not let all 1024 pixels vary**.

Instead:

1. Run a fast attribution method (**Grad-CAM++** or **DeepLIFT**) on the full image → produces a **heatmap**.
2. Use the heatmap to **select top-k** pixels (or top-k **SLIC superpixels**).
3. Run KernelSHAP only on those **k selected features**.

### Fixed-context game (important detail)

- Selected features **S (size k)** toggle between baseline and original depending on coalition mask.
- Non-selected features **Sᶜ** are always kept as **original** (never toggled).

So the SHAP game becomes: “What do the top-k features contribute **given that everything else stays fixed to the original image**?”

---

## Side-by-side pipeline (Normal vs Prior-Guided)

> Tip: GitHub renders this table correctly. If you paste into an editor that doesn't support Markdown tables, paste into GitHub directly (README.md) or a Markdown editor.

| Stage | Normal KernelSHAP (1024 pixels) | Prior-Guided KernelSHAP (Top-k) |
|---|---|---|
| Feature space | 1024 pixels (binary ON/OFF) | k features (pixels or superpixels) |
| Uses a prior heatmap? | No | Yes (Grad-CAM++ / DeepLIFT) |
| What toggles in coalitions | Any of the 1024 pixels | Only selected top-k toggle |
| What stays fixed | Nothing (all pixels may change) | Non-selected pixels always original |
| Masked image construction | `m ⊙ x + (1−m) ⊙ E_hwc` | Same idea, but `m` only controls selected features |
| Model score | Predicted-class logit | Predicted-class logit |
| Regression | WLS over 1024-D masks | WLS over k-D masks |
| Output φ | 1024 SHAP values | k SHAP values, scattered back (others = 0) |
| Compute cost | High | Much lower (k ≪ 1024) |
| Risk / limitation | — | May miss interactions outside top-k |

---

## Why the prior approach can fail sometimes (important)

Even if a pixel looks “unimportant” by itself (low heatmap score), it can become important **when combined with other pixels**.

Example intuition:

- Pixel A alone: not important  
- Pixel B alone: not important  
- **A + B together:** very important (interaction)

If either A or B was not included in top-k, that coalition is never explored.

✅ This method is often good and much faster  
❌ But it is **not guaranteed** to match full KernelSHAP

---

## Practical tips

- **Tune `k` first**:
  - smaller k = faster but can miss interactions
  - larger k = closer to full SHAP, but can become unstable / slow
  - try `k = 300, 600, 900` and check stability

- **Then tune `nsamples`**:
  - increases coalition sampling inside the chosen k-game
  - helps until diminishing returns

- **Try superpixels (SLIC)**:
  - groups pixels into meaningful regions
  - reduces off-manifold artifacts from masking single pixels
  - reduces effective feature count

---

## One-paragraph summary

Normal KernelSHAP toggles all 1024 pixels to sample coalitions and solve a WLS regression to assign Shapley values, but it’s slow. Our method first computes a fast heatmap prior (Grad-CAM++ or DeepLIFT), keeps only the top-k pixels (or superpixels) as features, fixes the rest of the image to its original values, and then runs KernelSHAP only on those k features. It is much faster and often captures the main explanation signal, but it can miss important coalitions involving features that were not selected.

---
