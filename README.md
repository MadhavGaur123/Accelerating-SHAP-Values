1) What “normal” KernelSHAP does (for an image)

Think of a 32×32 image as 1024 binary features (each pixel can be ON = original, or OFF = replaced by a baseline like the average image E_hwc).

To explain the model’s score for one class (we use the class logit), KernelSHAP:

Samples coalitions of pixels (which pixels are ON vs OFF).

Builds masked images for those coalitions: image(mask) = mask * image + (1 - mask) * baseline.

Evaluates the model on each masked image to get a score.

Solves a weighted least squares (WLS) problem to assign a Shapley value to each pixel, so that the sum of all pixel values roughly matches f(full image) − f(baseline).

This is faithful but expensive when you have 1024 features.

2) Our fast idea: use a prior to shrink the feature space

We keep the same SHAP logic, but we don’t let all 1024 pixels vary. Instead, we:

Run a quick attribution method on the full image, like Grad-CAM++ (or DeepLIFT).
This gives a heatmap telling us which areas look important.

Pick the top-k pixels (or top-k SLIC superpixels) based on that heatmap.
These are the only features we’ll toggle in the SHAP game.

Fix all other pixels to be as in the original image (always ON).
Only the selected top-k are switched ON/OFF against the baseline during SHAP sampling.

Run KernelSHAP on this k-feature game:

background = all k OFF (baseline on those k),

x_full = all k ON (original on those k),

sample coalitions among the k features and solve WLS to get k Shapley values.

Scatter back the k values to the 1024 grid (pixels not in top-k get 0).

This makes SHAP much faster because we solve a k-dimensional problem (k ≪ 1024).

3) The logic behind this

Full KernelSHAP is slow because there are many features (1024) and many possible coalitions.

A fast prior (Grad-CAM++ or DeepLIFT) is cheap and gives a reasonable guess about where the action is.

By only letting those k important features vary, we can compute Shapley values quickly while still capturing most of the explanation signal.

4) What this does not guarantee

A pixel that looks “unimportant” alone (and got excluded) might become important in combination with some selected pixels.
Because we fixed non-selected pixels to “original,” we don’t see coalitions that would have toggled them.

So this approach is an approximation: fast and often good, but not perfect.

5) Practical tips (plain English)

Choose k first. Small k = faster but may miss interactions; very large k = closer to full SHAP but can get unstable/slow.
Try a few values (e.g., 300, 600, 900) and see which gives stable, sensible maps.

nsamples (number of coalitions you sample) helps solve the chosen k-game better, but won’t fix a bad k.
After a point, more nsamples gives little extra benefit.

If you can, try superpixels (SLIC) instead of raw pixels—grouped regions often behave more naturally and reduce artifacts.

6) One-paragraph summary

Normal KernelSHAP randomly toggles all 1024 pixels and solves a regression to assign each pixel a contribution, but it’s slow. Our method first gets a heatmap prior (Grad-CAM++/DeepLIFT), keeps only the top-k pixels (or superpixels) as features, fixes everything else to the original image, and then runs KernelSHAP just on those k features. It’s the same SHAP machinery, just on a smaller, prior-guided feature set. It’s faster and usually good—but it can miss coalitions involving features we didn’t select.
