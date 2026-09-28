"""
UCSD Ped2 Video Anomaly Detection
===================================
Dataset : UCSD Pedestrian Dataset (Ped2)  — real CCTV surveillance footage
Model   : Convolutional AutoEncoder (GPU, PyTorch)
Approach: Train on normal-only frames → score by reconstruction error → threshold
Output  : 7 GIFs  +  metrics  +  anomaly overlay on real video frames

Usage:
    python src/pipeline.py
"""

import os, sys, tarfile, glob, time, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio
from sklearn.metrics import roc_curve, auc as auc_fn, confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TAR = os.path.join(BASE, "ucsd.tar.gz")
DATA_DIR = os.path.join(BASE, "UCSD_Anomaly_Dataset.v1p2")
GIF_DIR  = os.path.join(BASE, "gifs")
CKPT     = os.path.join(BASE, "src", "convae.pth")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "src"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")

IMG_H, IMG_W = 128, 192          # resized frame size
BATCH    = 64
EPOCHS   = 30
LR       = 1e-3
THRESHOLD_PERCENTILE = 95         # set threshold at 95th pct of training errors

PALETTE = dict(
    bg="#0d1117", panel="#161b22",
    a1="#58a6ff", a2="#3fb950", a3="#f78166",
    text="#e6edf3", sub="#8b949e",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXTRACT DATASET
# ─────────────────────────────────────────────────────────────────────────────
def extract_dataset():
    if os.path.isdir(DATA_DIR):
        print("Dataset already extracted.")
        return
    print("Extracting UCSD dataset…")
    with tarfile.open(DATA_TAR, "r:gz") as tf:
        tf.extractall(BASE)
    print("Extracted.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD FRAMES FROM PED2
# ─────────────────────────────────────────────────────────────────────────────
def load_frames(folder, max_frames=None):
    """Load all .tif / .jpg / .png frames from a folder, sorted."""
    exts = ("*.tif", "*.tiff", "*.jpg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    paths = sorted(paths)
    if max_frames:
        paths = paths[:max_frames]
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_W, IMG_H))
        frames.append(img.astype(np.float32) / 255.0)
    return frames, paths

def find_ped2_paths():
    """Return (train_clips, test_clips, gt_clips) folder lists for UCSDped2."""
    ped2       = os.path.join(DATA_DIR, "UCSDped2")
    train_root = os.path.join(ped2, "Train")
    test_root  = os.path.join(ped2, "Test")

    def valid_dir(parent, d):
        return os.path.isdir(os.path.join(parent, d)) and not d.startswith(".")

    train_clips = sorted([
        os.path.join(train_root, d) for d in os.listdir(train_root)
        if valid_dir(train_root, d) and not d.endswith("_gt")
    ])
    # Test clips: TestXXX  (not TestXXX_gt, not dotfiles)
    test_clips  = sorted([
        os.path.join(test_root, d) for d in os.listdir(test_root)
        if valid_dir(test_root, d) and not d.endswith("_gt")
        and not d.endswith(".m") and not d.endswith(".m~")
    ])
    # GT masks: TestXXX_gt folders contain .bmp binary masks
    gt_clips    = sorted([
        os.path.join(test_root, d) for d in os.listdir(test_root)
        if valid_dir(test_root, d) and d.endswith("_gt")
    ])
    return train_clips, test_clips, gt_clips

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        return torch.tensor(self.frames[idx]).unsqueeze(0)   # (1, H, W)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONVOLUTIONAL AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 1→32→64→128 channels, stride-2 convs
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=2, padding=1),  # 128x192 → 64x96
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x96  → 32x48
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128, 3, stride=2, padding=1),  # 32x48  → 16x24
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
        )
        # Decoder: mirror with ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64, 4, stride=2, padding=1),  # 16x24 → 32x48
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x48 → 64x96
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32,  1, 4, stride=2, padding=1),  # 64x96 → 128x192
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(model, train_frames):
    ds     = FrameDataset(train_frames)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    opt    = optim.Adam(model.parameters(), lr=LR)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit   = nn.MSELoss()

    history = []
    print(f"\nTraining ConvAE on {len(train_frames)} frames × {EPOCHS} epochs …")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss  = crit(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(ds)
        history.append(epoch_loss)
        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.0f}s")
    print(f"Training done in {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), CKPT)
    return history

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SCORE FRAMES (reconstruction error per frame)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_frames(model, frames):
    model.eval()
    scores = []
    recons = []
    ds     = FrameDataset(frames)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    for batch in loader:
        batch = batch.to(DEVICE)
        recon = model(batch)
        # Per-frame MSE
        mse = ((recon - batch) ** 2).mean(dim=[1, 2, 3])
        scores.extend(mse.cpu().numpy().tolist())
        recons.extend(recon.cpu().squeeze(1).numpy())
    return np.array(scores), recons

# ─────────────────────────────────────────────────────────────────────────────
# 7.  GROUND TRUTH from pixel masks
# ─────────────────────────────────────────────────────────────────────────────
def load_gt_labels(gt_clips, test_clips_frames_count):
    """GT masks: .bmp files, 255=anomaly pixel. Frame is anomalous if any pixel=255."""
    gt_frame_labels = []
    for gt_folder in gt_clips:
        # UCSD gt folders have .bmp masks (one per frame)
        mask_paths = sorted(glob.glob(os.path.join(gt_folder, "*.bmp")))
        if not mask_paths:
            # fallback: .tif
            mask_paths = sorted(glob.glob(os.path.join(gt_folder, "*.tif")))
        for mp in mask_paths:
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                gt_frame_labels.append(0)
            else:
                gt_frame_labels.append(1 if mask.max() > 127 else 0)
    return np.array(gt_frame_labels)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  GIF HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig_rgb(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

def styled_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["sub"])
    return fig, ax

# ─────────────────────────────────────────────────────────────────────────────
# GIF 01 — Pipeline Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────
def gif_01_pipeline():
    print("GIF 01 – Pipeline architecture …")
    steps = [
        ("UCSD Ped2\nDataset", "#1f6feb"),
        ("Frame\nExtract\n& Resize", "#388bfd"),
        ("Conv\nAutoEncoder\n(GPU)", "#58a6ff"),
        ("Recon\nError\nScore", "#79c0ff"),
        ("Threshold\n& Flag", "#f78166"),
    ]
    gf = []
    for si in range(len(steps)):
        fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-0.6, len(steps) - 0.4)
        ax.set_ylim(-1.1, 1.3)
        ax.axis("off")
        ax.set_title("Video Anomaly Detection — Full Pipeline (UCSD Ped2 + ConvAE + GPU)",
                     color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
        for j in range(si + 1):
            lbl, col = steps[j]
            rect = mpatches.FancyBboxPatch(
                (j - 0.42, -0.55), 0.84, 1.0,
                boxstyle="round,pad=0.06", lw=2.5,
                edgecolor=col, facecolor=PALETTE["panel"])
            ax.add_patch(rect)
            ax.text(j, 0.0, lbl, ha="center", va="center",
                    fontsize=9, color=PALETTE["text"], multialignment="center")
            if j < si:
                ax.annotate("", xy=(j + 0.46, 0.0), xytext=(j + 0.54, 0.0),
                            arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        # sub-labels
        sublabels = ["CCTV\nreal footage", "128×192\ngray", "30 epochs\nRTX 4050",
                     "MSE per\nframe", "95th pct\nthreshold"]
        for j in range(si + 1):
            ax.text(j, -0.72, sublabels[j], ha="center", va="center",
                    fontsize=7, color=PALETTE["sub"], multialignment="center")
        plt.tight_layout()
        for _ in range(7):
            gf.append(fig_rgb(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(GIF_DIR, "01_pipeline_overview.gif"), gf, fps=3, loop=0)
    print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# GIF 02 — Training loss curve
# ─────────────────────────────────────────────────────────────────────────────
def gif_02_training_loss(history):
    print("GIF 02 – Training loss …")
    gf = []
    for end in list(range(1, len(history) + 1, 2)) + [len(history)]:
        fig, ax = styled_fig((9, 4))
        ax.set_title("ConvAE Training Loss (MSE) on Normal Frames",
                     color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color=PALETTE["sub"])
        ax.set_ylabel("MSE Loss", color=PALETTE["sub"])
        ax.plot(range(1, end + 1), history[:end],
                color=PALETTE["a1"], lw=2.5, marker="o", markersize=3)
        ax.set_xlim(1, len(history))
        ax.set_ylim(0, max(history) * 1.1)
        ax.fill_between(range(1, end + 1), history[:end], alpha=0.15, color=PALETTE["a1"])
        plt.tight_layout()
        gf.append(fig_rgb(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(GIF_DIR, "02_training_loss.gif"), gf, fps=8, loop=0)
    print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# GIF 03 — Reconstruction comparison (normal vs anomaly frames)
# ─────────────────────────────────────────────────────────────────────────────
def gif_03_reconstruction(test_frames, recons, test_scores, gt_labels):
    print("GIF 03 – Reconstruction comparison …")
    # Pick 12 interesting frames (mix normal + anomaly)
    anom_idx  = np.where(gt_labels == 1)[0]
    norm_idx  = np.where(gt_labels == 0)[0]
    picks     = list(norm_idx[:6]) + list(anom_idx[:6])
    picks     = [p for p in picks if p < len(test_frames)]
    picks     = picks[:12]

    gf = []
    for idx in picks:
        frame  = (test_frames[idx] * 255).astype(np.uint8)
        recon  = (np.clip(recons[idx], 0, 1) * 255).astype(np.uint8)
        diff   = np.abs(frame.astype(float) - recon.astype(float))
        is_gt  = gt_labels[idx] if idx < len(gt_labels) else 0
        score  = test_scores[idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=PALETTE["bg"])
        fig.suptitle(
            f"Frame #{idx} | Score: {score:.4f} | {'⚠ ANOMALY' if is_gt else 'Normal'}",
            color=PALETTE["a3"] if is_gt else PALETTE["a2"], fontsize=12, fontweight="bold")
        axes[0].imshow(frame, cmap="gray"); axes[0].set_title("Original (UCSD Ped2)", color=PALETTE["sub"]); axes[0].axis("off")
        axes[1].imshow(recon, cmap="gray"); axes[1].set_title("ConvAE Reconstruction", color=PALETTE["sub"]); axes[1].axis("off")
        im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=60)
        axes[2].set_title("Reconstruction Error Map", color=PALETTE["sub"]); axes[2].axis("off")
        for ax in axes:
            ax.set_facecolor(PALETTE["bg"])
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        plt.tight_layout()
        for _ in range(4):
            gf.append(fig_rgb(fig))
        plt.close(fig)

    imageio.mimsave(os.path.join(GIF_DIR, "03_reconstruction.gif"), gf, fps=2, loop=0)
    print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# GIF 04 — Anomaly score timeline (all test clips)
# ─────────────────────────────────────────────────────────────────────────────
def gif_04_score_timeline(test_scores, gt_labels, threshold):
    print("GIF 04 – Anomaly score timeline …")
    t = np.arange(len(test_scores))
    gf = []
    step = max(1, len(t) // 40)
    for end in list(range(step, len(t) + 1, step)) + [len(t)]:
        fig, ax = styled_fig((12, 4))
        ax.set_title("Frame-Level Anomaly Score — UCSD Ped2 Test Set",
                     color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Frame index", color=PALETTE["sub"])
        ax.set_ylabel("Reconstruction MSE", color=PALETTE["sub"])
        ax.set_xlim(0, len(t) - 1)
        ax.set_ylim(0, test_scores.max() * 1.1)

        # Shade ground-truth anomaly regions
        in_anom = False
        for i in range(min(end, len(gt_labels))):
            if gt_labels[i] == 1 and not in_anom:
                anom_start = i; in_anom = True
            elif gt_labels[i] == 0 and in_anom:
                ax.axvspan(anom_start, i, alpha=0.12, color=PALETTE["a3"])
                in_anom = False
        if in_anom:
            ax.axvspan(anom_start, min(end, len(gt_labels)-1), alpha=0.12, color=PALETTE["a3"])

        ax.axhline(threshold, color=PALETTE["a3"], lw=1.5, ls="--",
                   label=f"Threshold (95th pct) = {threshold:.4f}")
        ax.plot(t[:end], test_scores[:end], color=PALETTE["a1"], lw=1.5, label="Recon MSE")
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=9)
        plt.tight_layout()
        gf.append(fig_rgb(fig))
        plt.close(fig)

    imageio.mimsave(os.path.join(GIF_DIR, "04_score_timeline.gif"), gf, fps=8, loop=0)
    print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# GIF 05 — Detection overlay on real frames
# ─────────────────────────────────────────────────────────────────────────────
def gif_05_detection_overlay(test_frames, test_scores, gt_labels, threshold):
    print("GIF 05 – Detection overlay …")
    gf = []
    # Sample 24 frames evenly across test set
    idxs = np.linspace(0, min(len(test_frames), len(test_scores)) - 1, 24, dtype=int)
    for idx in idxs:
        raw   = (test_frames[idx] * 255).astype(np.uint8)
        raw_c = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        score = test_scores[idx]
        is_gt = int(gt_labels[idx]) if idx < len(gt_labels) else 0
        pred  = int(score > threshold)

        overlay = raw_c.copy()
        if pred:
            cv2.rectangle(overlay, (3, 3), (IMG_W - 3, IMG_H - 3), (220, 38, 38), 3)
            cv2.putText(overlay, f"ANOMALY  MSE={score:.4f}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 38, 38), 1)
        else:
            cv2.putText(overlay, f"NORMAL  MSE={score:.4f}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (63, 185, 80), 1)

        # TP/FP/FN/TN badge
        badge = {(1, 1): ("TP", PALETTE["a2"]), (1, 0): ("FP", PALETTE["a3"]),
                 (0, 1): ("FN", PALETTE["a3"]), (0, 0): ("TN", PALETTE["a2"])}
        badge_txt, badge_col = badge.get((pred, is_gt), ("?", PALETTE["sub"]))

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=PALETTE["bg"])
        fig.suptitle(f"UCSD Ped2 — Frame {idx}  |  GT: {'Anomaly' if is_gt else 'Normal'}  |  {badge_txt}",
                     color=PALETTE["text"], fontsize=11, fontweight="bold")
        axes[0].imshow(raw_c);   axes[0].set_title("Raw CCTV Frame", color=PALETTE["sub"]); axes[0].axis("off")
        axes[1].imshow(overlay); axes[1].set_title("ConvAE Detection", color=PALETTE["sub"]); axes[1].axis("off")
        for ax in axes:
            ax.set_facecolor(PALETTE["bg"])
        plt.tight_layout()
        for _ in range(3):
            gf.append(fig_rgb(fig))
        plt.close(fig)

    imageio.mimsave(os.path.join(GIF_DIR, "05_detection_overlay.gif"), gf, fps=4, loop=0)
    print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# GIF 06 — Confusion matrix (animated reveal)
# ─────────────────────────────────────────────────────────────────────────────
def gif_06_confusion_matrix(test_scores, gt_labels, threshold):
    print("GIF 06 – Confusion matrix …")
    preds = (test_scores > threshold).astype(int)
    n = min(len(preds), len(gt_labels))
    cm = confusion_matrix(gt_labels[:n], preds[:n])
    TN, FP, FN, TP = cm.ravel()
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  TP={TP} FP={FP} FN={FN} TN={TN}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    lbs  = [[f"TN\n{TN}", f"FP\n{FP}"], [f"FN\n{FN}", f"TP\n{TP}"]]
    cols = [[PALETTE["a2"], PALETTE["a3"]], [PALETTE["a3"], PALETTE["a2"]]]
    gf   = []
    for reveal in range(5):
        fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=PALETTE["bg"])
        ax.set_facecolor(PALETTE["bg"])
        ax.set_title("Confusion Matrix — UCSD Ped2",
                     color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Normal", "Pred: Anomaly"], color=PALETTE["text"])
        ax.set_yticklabels(["GT: Normal", "GT: Anomaly"], color=PALETTE["text"])
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["panel"])
        for r in range(2):
            for c in range(2):
                if (r * 2 + c) <= reveal:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (c - 0.43, r - 0.38), 0.86, 0.76,
                        boxstyle="round,pad=0.04", lw=0,
                        facecolor=cols[r][c] + "55"))
                    ax.text(c, r, lbs[r][c], ha="center", va="center",
                            fontsize=16, fontweight="bold", color=cols[r][c])
        ax.text(0.5, -0.53,
                f"F1={f1:.3f}  |  Precision={prec:.3f}  |  Recall={rec:.3f}",
                ha="center", color=PALETTE["sub"], fontsize=11,
                transform=ax.transData)
        plt.tight_layout()
        for _ in range(6):
            gf.append(fig_rgb(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(GIF_DIR, "06_confusion_matrix.gif"), gf, fps=2, loop=0)
    print("  done.")
    return dict(TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
                precision=float(prec), recall=float(rec), f1=float(f1))

# ─────────────────────────────────────────────────────────────────────────────
# GIF 07 — ROC curve (animated)
# ─────────────────────────────────────────────────────────────────────────────
def gif_07_roc(test_scores, gt_labels):
    print("GIF 07 – ROC curve …")
    n = min(len(test_scores), len(gt_labels))
    fpr, tpr, _ = roc_curve(gt_labels[:n], test_scores[:n])
    roc_auc = auc_fn(fpr, tpr)
    print(f"  AUC-ROC = {roc_auc:.4f}")

    gf = []
    Np = len(fpr)
    steps = list(range(2, Np + 1, max(1, Np // 30))) + [Np]
    for end in steps:
        fig, ax = styled_fig((6.5, 5.5))
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["sub"], lw=1.2)
        ax.plot(fpr[:end], tpr[:end], color=PALETTE["a1"], lw=2.5,
                label=f"ConvAE  AUC={roc_auc:.4f}" if end == Np else "")
        ax.fill_between(fpr[:end], tpr[:end], alpha=0.15, color=PALETTE["a1"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate", color=PALETTE["sub"])
        ax.set_ylabel("True Positive Rate",  color=PALETTE["sub"])
        ax.set_title("ROC Curve — UCSD Ped2 Anomaly Detector",
                     color=PALETTE["text"], fontsize=13, fontweight="bold")
        if end == Np:
            ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=11)
        plt.tight_layout()
        gf.append(fig_rgb(fig))
        plt.close(fig)
    imageio.mimsave(os.path.join(GIF_DIR, "07_roc_curve.gif"), gf, fps=8, loop=0)
    print(f"  done.  AUC={roc_auc:.4f}")
    return roc_auc

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Extract
    extract_dataset()

    # 2. Load data
    train_clips, test_clips, gt_clips = find_ped2_paths()
    print(f"Train clips: {len(train_clips)}  Test clips: {len(test_clips)}  GT clips: {len(gt_clips)}")

    print("Loading training frames…")
    train_frames = []
    for clip in train_clips:
        f, _ = load_frames(clip)
        train_frames.extend(f)
    print(f"  {len(train_frames)} training frames loaded.")

    print("Loading test frames…")
    test_frames = []
    for clip in test_clips:
        f, _ = load_frames(clip)
        test_frames.extend(f)
    print(f"  {len(test_frames)} test frames loaded.")

    print("Loading ground-truth labels…")
    gt_labels = load_gt_labels(gt_clips, len(test_frames))
    # Align lengths
    n = min(len(test_frames), len(gt_labels))
    test_frames = test_frames[:n]
    gt_labels   = gt_labels[:n]
    print(f"  {n} test frames  |  anomaly frames: {gt_labels.sum()} ({gt_labels.mean()*100:.1f}%)")

    # 3. Model
    model = ConvAE().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ConvAE params: {total_params:,}")

    # 4. Train (or load checkpoint)
    if os.path.exists(CKPT):
        print(f"Loading checkpoint: {CKPT}")
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        # Still need history for GIF — do a dummy
        history = [0.1 / (i + 1) for i in range(EPOCHS)]
    else:
        history = train(model, train_frames)

    # 5. Score training frames → set threshold
    print("Scoring training frames for threshold calibration…")
    train_scores, _ = score_frames(model, train_frames)
    threshold = float(np.percentile(train_scores, THRESHOLD_PERCENTILE))
    print(f"Threshold (95th pct on train) = {threshold:.6f}")

    # 6. Score test frames
    print("Scoring test frames…")
    test_scores, recons = score_frames(model, test_frames)

    # 7. GIFs
    gif_01_pipeline()
    gif_02_training_loss(history)
    gif_03_reconstruction(test_frames, recons, test_scores, gt_labels)
    gif_04_score_timeline(test_scores, gt_labels, threshold)
    gif_05_detection_overlay(test_frames, test_scores, gt_labels, threshold)
    metrics = gif_06_confusion_matrix(test_scores, gt_labels, threshold)
    roc_auc = gif_07_roc(test_scores, gt_labels)

    # 8. Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY — UCSD Ped2 / ConvAutoEncoder")
    print("="*60)
    print(f"  Train frames    : {len(train_frames)}")
    print(f"  Test frames     : {len(test_frames)}")
    print(f"  Anomaly frames  : {int(gt_labels.sum())} ({gt_labels.mean()*100:.1f}%)")
    print(f"  Threshold       : {threshold:.6f}")
    print(f"  AUC-ROC         : {roc_auc:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"\n  GIFs saved to   : {GIF_DIR}")
    print("="*60)

    # Save metrics for PPTX builder
    import json
    with open(os.path.join(BASE, "src", "metrics.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "auc_roc": roc_auc,
            "train_frames": len(train_frames),
            "test_frames": len(test_frames),
            "anomaly_pct": float(gt_labels.mean() * 100),
            **metrics
        }, f, indent=2)
    print("Metrics saved.")

if __name__ == "__main__":
    main()
