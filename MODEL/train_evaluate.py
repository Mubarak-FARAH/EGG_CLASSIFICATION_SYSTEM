"""
  Model    : EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
  Dataset  : FINAL_DATASET-V2  (train / validation / test)
  Outputs  : saved model, training curves, confusion matrix, PR curves,
             mAP metrics, Grad-CAM heatmaps, LIME explanations, SHAP values
  Install dependencies:
      pip install torch torchvision timm scikit-learn matplotlib seaborn
                  grad-cam lime shap numpy pandas pillow tqdm
"""

# Imports
import os, random, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score,
)
from sklearn.preprocessing import label_binarize

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# LIME
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries

# SHAP
import shap

warnings.filterwarnings("ignore")

# Configuration
CFG = dict(
    dataset_root   = Path("FINAL_DATASET-V2"),
    output_dir     = Path("RESULTS"),

    # Model
    model_name     = "efficientnet_b0",   # any timm model works here
    num_epochs     = 30,
    batch_size     = 32,
    learning_rate  = 1e-4,
    weight_decay   = 1e-4,
    patience       = 7,                   # early-stopping patience
    img_size       = 224,

    # Evaluation
    gradcam_samples = 4,                  # images per class for Grad-CAM
    lime_samples    = 6,                  # total LIME explanation images
    shap_samples    = 50,                 # background samples for SHAP

    seed           = 42,
)

# Reproducibility
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

CFG["output_dir"].mkdir(parents=True, exist_ok=True)

# Dataset: label is the prefix before the first "__"  in the filename
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

TRAIN_TF = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

EVAL_TF = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class BirdDataset(Dataset):
    def __init__(self, split: str, class_to_idx: dict = None, transform=None):
        self.transform    = transform
        self.split_dir    = CFG["dataset_root"] / split
        self.samples      = []          # (path, label_str)
        self.class_to_idx = class_to_idx or {}

        for fpath in sorted(self.split_dir.iterdir()):
            if fpath.suffix.lower() not in IMG_EXTS:
                continue
            label = fpath.name.split("__")[0]
            self.samples.append((fpath, label))

        # Build class map from training split if not provided
        if not self.class_to_idx:
            classes = sorted({s[1] for s in self.samples})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        img  = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label_str]


def build_loaders():
    train_ds = BirdDataset("train",      transform=TRAIN_TF)
    val_ds   = BirdDataset("validation", class_to_idx=train_ds.class_to_idx, transform=EVAL_TF)
    test_ds  = BirdDataset("test",       class_to_idx=train_ds.class_to_idx, transform=EVAL_TF)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print(f"Classes  : {len(train_ds.class_to_idx)}")
    print(f"Train    : {len(train_ds)} images")
    print(f"Val      : {len(val_ds)}   images")
    print(f"Test     : {len(test_ds)}  images")

    return train_loader, val_loader, test_loader, train_ds.class_to_idx, train_ds.idx_to_class


# Model

def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model(
        CFG["model_name"],
        pretrained=True,
        num_classes=num_classes,
    )
    return model.to(DEVICE)


# Training Loop

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def train(model, train_loader, val_loader, num_classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["learning_rate"],
                            weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["num_epochs"])

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val  = 0.0
    patience_ctr = 0
    best_path = CFG["output_dir"] / "best_model.pth"

    print("\n━━━  Training  ━━━")
    for epoch in range(1, CFG["num_epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_val:
            best_val    = vl_acc
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            flag = "  ✓ saved"
        else:
            patience_ctr += 1

        print(f"  Epoch {epoch:3d}/{CFG['num_epochs']}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{flag}")

        if patience_ctr >= CFG["patience"]:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # Reload best weights
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"\n  Best val accuracy: {best_val:.4f}  →  {best_path}")
    return history


# Plot helpers

def save_fig(name: str):
    path = CFG["output_dir"] / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#E74C3C")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498DB")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#E74C3C")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="#3498DB")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    save_fig("01_training_curves")


# Confusion Matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(28, 22))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.4, ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    save_fig("02_confusion_matrix")


# Precision–Recall Curves + mAP

def compute_map_iou(y_true, y_probs, class_names):
    """
    mAP@0.5  : average precision across classes at IoU / threshold ≥ 0.5
    mAP@0.5:0.95 : mean of APs computed at thresholds 0.50, 0.55, …, 0.95
    IoU (soft) : mean diagonal of the soft confusion matrix
    """
    n_classes  = len(class_names)
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))
    ap_scores  = {}

    for c in range(n_classes):
        ap_scores[c] = average_precision_score(y_bin[:, c], y_probs[:, c])

    # mAP@0.5
    map50 = float(np.mean(list(ap_scores.values())))

    # mAP@0.5:0.95
    thresholds = np.arange(0.50, 1.00, 0.05)
    ap_at_thresh = []
    for thr in thresholds:
        preds_at_thr = (y_probs >= thr).astype(int)
        per_class = []
        for c in range(n_classes):
            if y_bin[:, c].sum() == 0:
                continue
            per_class.append(average_precision_score(y_bin[:, c], preds_at_thr[:, c]))
        ap_at_thresh.append(np.mean(per_class) if per_class else 0.0)
    map50_95 = float(np.mean(ap_at_thresh))

    # Soft IoU
    y_true_arr = np.array(y_true)
    iou_vals = []
    for i, true_cls in enumerate(y_true_arr):
        p_true  = y_probs[i, true_cls]
        p_other = 1.0 - p_true
        iou_vals.append(p_true / (1.0 + p_other + 1e-8))
    mean_iou = float(np.mean(iou_vals))

    print(f"\n  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.5:0.95 : {map50_95:.4f}")
    print(f"  Mean IoU     : {mean_iou:.4f}")

    return ap_scores, map50, map50_95, mean_iou


def plot_pr_curves(y_true, y_probs, class_names, ap_scores):
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    # Per-class PR curves
    cols, rows = 3, 3
    per_page   = cols * rows
    n_pages    = int(np.ceil(n_classes / per_page))

    for page in range(n_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 13))
        fig.suptitle(f"Precision–Recall Curves (page {page+1}/{n_pages})",
                     fontsize=13, fontweight="bold")
        axes = axes.flatten()

        for j, ax in enumerate(axes):
            c = page * per_page + j
            if c >= n_classes:
                ax.axis("off"); continue
            prec, rec, _ = precision_recall_curve(y_bin[:, c], y_probs[:, c])
            ax.plot(rec, prec, color="#3498DB", lw=1.8)
            ax.fill_between(rec, prec, alpha=0.15, color="#3498DB")
            ax.set_title(f"{class_names[c]}\nAP={ap_scores[c]:.3f}",
                         fontsize=8, fontweight="bold")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
            ax.set_xlabel("Recall", fontsize=7)
            ax.set_ylabel("Precision", fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig(f"03_pr_curves_page{page+1}")

    # Macro-average PR curve
    all_prec = dict(); all_rec = dict()
    for c in range(n_classes):
        all_prec[c], all_rec[c], _ = precision_recall_curve(y_bin[:, c], y_probs[:, c])

    mean_rec   = np.linspace(0, 1, 200)
    mean_prec  = np.zeros_like(mean_rec)
    for c in range(n_classes):
        mean_prec += np.interp(mean_rec, all_rec[c][::-1], all_prec[c][::-1])
    mean_prec /= n_classes

    macro_ap = float(np.mean(list(ap_scores.values())))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_rec, mean_prec, color="#E74C3C", lw=2,
            label=f"Macro-avg (AP={macro_ap:.3f})")
    ax.fill_between(mean_rec, mean_prec, alpha=0.15, color="#E74C3C")
    ax.set_title("Macro-Average Precision–Recall Curve",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig("04_pr_macro_avg")


# Evaluation — Per-class metrics bar chart

def plot_per_class_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    rows = [(cls, report[cls]["precision"],
                  report[cls]["recall"],
                  report[cls]["f1-score"])
            for cls in class_names if cls in report]

    df = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1"])
    df = df.sort_values("F1")

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.35)))
    y_pos = np.arange(len(df))
    bar_w = 0.26
    ax.barh(y_pos - bar_w, df["Precision"], bar_w, label="Precision", color="#3498DB")
    ax.barh(y_pos,         df["Recall"],    bar_w, label="Recall",    color="#2ECC71")
    ax.barh(y_pos + bar_w, df["F1"],        bar_w, label="F1-Score",  color="#E74C3C")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Class"], fontsize=8)
    ax.set_xlabel("Score"); ax.set_xlim(0, 1.05)
    ax.set_title("Per-Class Metrics", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_fig("05_per_class_metrics")
    return report


def save_metrics_summary(report, map50, map50_95, mean_iou):
    summary = {
        "overall_accuracy"  : report["accuracy"],
        "macro_precision"   : report["macro avg"]["precision"],
        "macro_recall"      : report["macro avg"]["recall"],
        "macro_f1"          : report["macro avg"]["f1-score"],
        "weighted_f1"       : report["weighted avg"]["f1-score"],
        "mAP_at_0.5"        : map50,
        "mAP_at_0.5_to_0.95": map50_95,
        "mean_IoU"          : mean_iou,
    }
    out_path = CFG["output_dir"] / "metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Metrics summary → {out_path}")
    for k, v in summary.items():
        print(f"    {k:<28}: {v:.4f}")


# Grad-CAM

def run_gradcam(model, test_ds, idx_to_class, n_samples=4):
    print("\n━━━  Grad-CAM  ━━━")

    # Identify the last conv layer (EfficientNet-B0)
    target_layers = [model.conv_head]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Sample n_samples random images
    indices = random.sample(range(len(test_ds)), min(n_samples, len(test_ds)))
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )

    cols = min(4, n_samples)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 7))
    fig.suptitle("Grad-CAM Heatmaps", fontsize=14, fontweight="bold")

    axes = axes.flatten() if n_samples > 1 else [axes]

    for plot_i, ds_idx in enumerate(indices):
        img_tensor, label_idx = test_ds[ds_idx]
        input_batch = img_tensor.unsqueeze(0).to(DEVICE)

        grayscale_cam = cam(input_tensor=input_batch)[0]

        # Denormalise for display
        img_display = inv_norm(img_tensor).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1).astype(np.float32)

        cam_image = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

        true_label = idx_to_class[label_idx]
        with torch.no_grad():
            pred_idx = model(input_batch).argmax(1).item()
        pred_label = idx_to_class[pred_idx]

        row_offset = (plot_i // cols) * 2 * cols
        col        = plot_i % cols

        ax_orig = axes[row_offset + col]
        ax_cam  = axes[row_offset + cols + col]

        ax_orig.imshow(img_display)
        ax_orig.set_title(f"True: {true_label}", fontsize=7)
        ax_orig.axis("off")

        ax_cam.imshow(cam_image)
        ax_cam.set_title(f"Pred: {pred_label}", fontsize=7,
                         color="green" if pred_label == true_label else "red")
        ax_cam.axis("off")

    plt.tight_layout()
    save_fig("06_gradcam")


# LIME

def run_lime(model, test_ds, idx_to_class, n_samples=6):
    print("\n━━━  LIME  ━━━")
    model.eval()

    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )
    to_tensor = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def predict_fn(images_np):
        """images_np : (N, H, W, 3) uint8."""
        batch = torch.stack([
            to_tensor(Image.fromarray(img.astype(np.uint8)))
            for img in images_np
        ]).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        return probs

    explainer = lime.lime_image.LimeImageExplainer()
    indices   = random.sample(range(len(test_ds)), min(n_samples, len(test_ds)))

    cols = min(3, n_samples)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle("LIME Explanations", fontsize=14, fontweight="bold")
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for plot_i, ds_idx in enumerate(indices):
        img_tensor, label_idx = test_ds[ds_idx]
        img_display = inv_norm(img_tensor).permute(1, 2, 0).numpy()
        img_display = (np.clip(img_display, 0, 1) * 255).astype(np.uint8)

        explanation = explainer.explain_instance(
            img_display,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=300,
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )

        axes[plot_i].imshow(mark_boundaries(temp / 255.0, mask))
        axes[plot_i].set_title(
            f"True: {idx_to_class[label_idx]}\nExplained: {idx_to_class[top_label]}",
            fontsize=7,
        )
        axes[plot_i].axis("off")

    for ax in axes[n_samples:]:
        ax.axis("off")

    plt.tight_layout()
    save_fig("07_lime")


# SHAP

def _disable_inplace(model: nn.Module) -> nn.Module:
    """Set inplace=False on all activations — required for SHAP GradientExplainer
    compatibility with EfficientNet SiLU/ReLU inplace ops."""
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    return model


def _normalise_shap_values(shap_values, n_explain, n_classes):
    """
    Different SHAP / PyTorch versions return shap_values in different shapes:
      A) list of n_classes arrays, each (n_explain, C, H, W)  → index [cls][img]
      B) single array (n_explain, n_classes, C, H, W)         → index [img, cls]
      C) single array (n_explain, C, H, W)  (mean over classes)

    This function always returns a numpy array of shape
    (n_explain, n_classes_or_1, C, H, W) so downstream code indexes safely.
    """
    if isinstance(shap_values, list):
        # Case A — list of per-class arrays
        arr = np.stack(shap_values, axis=1)          # (n_explain, n_classes, C, H, W)
        return arr

    sv = np.array(shap_values)
    if sv.ndim == 5 and sv.shape[0] == n_explain:
        return sv                                     # Case B already correct
    if sv.ndim == 5 and sv.shape[0] == n_classes:
        return sv.transpose(1, 0, 2, 3, 4)           # swap axes to (n_explain, n_classes, C, H, W)
    if sv.ndim == 4:
        return sv[:, np.newaxis, ...]                 # Case C — add dummy class axis
    return sv


def run_shap(model, test_ds, idx_to_class, n_background=50, n_explain=4):
    print("\n━━━  SHAP  ━━━")

    _disable_inplace(model)
    model.eval()

    all_indices = list(range(len(test_ds)))
    random.shuffle(all_indices)

    bg_imgs   = torch.stack([test_ds[i][0] for i in all_indices[:n_background]]).to(DEVICE)
    exp_items = [(test_ds[i][0], test_ds[i][1]) for i in all_indices[n_background:n_background + n_explain]]
    exp_imgs  = torch.stack([x[0] for x in exp_items]).to(DEVICE)
    exp_lbls  = [x[1] for x in exp_items]

    explainer   = shap.GradientExplainer(model, bg_imgs)
    raw_sv      = explainer.shap_values(exp_imgs)

    n_classes   = len(idx_to_class)
    # Normalise to (n_explain, n_classes_or_1, C, H, W)
    sv_arr      = _normalise_shap_values(raw_sv, n_explain, n_classes)
    print(f"  SHAP values shape after normalisation: {sv_arr.shape}")

    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )

    show_classes = min(3, sv_arr.shape[1])   # cap at available class axes

    for img_i in range(n_explain):
        true_lbl = idx_to_class[exp_lbls[img_i]]
        fig, axes = plt.subplots(1, show_classes + 1,
                                 figsize=((show_classes + 1) * 4, 4))
        fig.suptitle(f"SHAP — Image {img_i+1}  (True: {true_lbl})",
                     fontsize=11, fontweight="bold")

        img_display = inv_norm(exp_imgs[img_i].cpu()).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        axes[0].imshow(img_display)
        axes[0].set_title("Original"); axes[0].axis("off")

        with torch.no_grad():
            probs     = torch.softmax(model(exp_imgs[img_i:img_i+1]), dim=1)[0].cpu().numpy()
        top_classes = np.argsort(probs)[::-1][:show_classes]

        for j, cls_idx in enumerate(top_classes):
            # Safe indexing: clamp cls_idx to available class axes
            sv_cls_idx = min(int(cls_idx), sv_arr.shape[1] - 1)
            sv         = sv_arr[img_i, sv_cls_idx]       # (C, H, W)
            sv_mean    = np.mean(np.abs(sv), axis=0)     # (H, W)
            sv_norm    = sv_mean / (sv_mean.max() + 1e-8)

            axes[j + 1].imshow(img_display)
            axes[j + 1].imshow(sv_norm, cmap="hot", alpha=0.5)
            axes[j + 1].set_title(
                f"{idx_to_class[cls_idx]}\n(p={probs[cls_idx]:.2f})",
                fontsize=7,
            )
            axes[j + 1].axis("off")

        plt.tight_layout()
        save_fig(f"08_shap_img{img_i+1}")


# Main

def main():
    print("  Bird Species Classification — Training & Evaluation")

    # Data
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = build_loaders()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes  = len(class_names)

    # Model
    model = build_model(num_classes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    #  Training 
    history = train(model, train_loader, val_loader, num_classes)
    plot_training_curves(history)

    #  Test Evaluation
    print("\n━━━  Evaluating on test split  ━━━")
    criterion = nn.CrossEntropyLoss()
    _, test_acc, y_pred, y_true, y_probs = evaluate(model, test_loader, criterion)
    print(f"  Test Accuracy: {test_acc:.4f}")

    #  Metrics
    plot_confusion_matrix(y_true, y_pred, class_names)
    ap_scores, map50, map50_95, mean_iou = compute_map_iou(y_true, y_probs, class_names)
    plot_pr_curves(y_true, y_probs, class_names, ap_scores)
    report = plot_per_class_metrics(y_true, y_pred, class_names)
    save_metrics_summary(report, map50, map50_95, mean_iou)

    #  Interpretability 
    test_ds = test_loader.dataset
    run_gradcam(model, test_ds, idx_to_class, n_samples=CFG["gradcam_samples"])
    run_lime(model, test_ds, idx_to_class, n_samples=CFG["lime_samples"])
    run_shap(model, test_ds, idx_to_class,
             n_background=CFG["shap_samples"], n_explain=4)

    #  Done 
    print("  All done!  Results saved to:", CFG["output_dir"].resolve())
    print("""
  Output files:
    best_model.pth              ← trained model weights
    metrics_summary.json        ← accuracy, F1, mAP, IoU
    01_training_curves.png      ← loss & accuracy over epochs
    02_confusion_matrix.png     ← counts + normalised heatmap
    03_pr_curves_page*.png      ← per-class Precision-Recall curves
    04_pr_macro_avg.png         ← macro-average PR curve
    05_per_class_metrics.png    ← precision / recall / F1 bar chart
    06_gradcam.png              ← Grad-CAM heatmaps
    07_lime.png                 ← LIME superpixel explanations
    08_shap_img*.png            ← SHAP attribution heatmaps
    """)


if __name__ == "__main__":
    main()