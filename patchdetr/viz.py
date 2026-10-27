"""Notebook-friendly visualisation helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .losses import cxcywh_to_xyxy


def denormalize(img: torch.Tensor) -> np.ndarray:
	mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
	return (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _draw(ax, xyxy: np.ndarray, color: str, labels: Optional[Sequence[str]] = None) -> None:
	for i, (x0, y0, x1, y1) in enumerate(xyxy):
		ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
		if labels is not None:
			ax.text(x0, y0 - 2, labels[i], color=color, fontsize=7)


@torch.no_grad()
def predict(model, sample: Dict[str, Any], device: str, score_thr: float = 0.5) -> Dict[str, Any]:
	"""Runs one dataset sample through the model; boxes are returned in pixel xyxy."""
	model.eval()
	out = model(sample["clean"][None].to(device), sample["motion"][None].to(device))
	scores = (1 - out["logits"][0].float().softmax(-1)[:, 1]).cpu()
	boxes = cxcywh_to_xyxy(out["boxes"][0].float().cpu())
	_, H, W = sample["clean"].shape
	scale = torch.tensor([W, H, W, H])
	keep = scores >= score_thr
	return {"scores": scores[keep], "boxes": (boxes[keep] * scale).numpy(), "out": out, "all_scores": scores}


def plot_predictions(model, dataset, indices: Sequence[int], device: str, score_thr: float = 0.5, cols: int = 2):
	"""Ground truth (green) vs predictions (red) on dataset samples (use a *_degraded dataset to
	see behaviour on mud-covered objects)."""
	rows = int(np.ceil(len(indices) / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.6 * rows), squeeze=False)
	for ax, i in zip(axes.flat, indices):
		s = dataset[i]
		p = predict(model, s, device, score_thr)
		_, H, W = s["clean"].shape
		ax.imshow(denormalize(s["clean"]))
		b = s["boxes"].numpy()
		if len(b):
			_draw(ax, cxcywh_to_xyxy(torch.from_numpy(b)).numpy() * np.array([W, H, W, H]), "lime")
		_draw(ax, p["boxes"], "red", [f"{x:.2f}" for x in p["scores"]])
		ax.set_title(f"{s['path'].split('/')[-1]}  GT={len(b)} pred={len(p['scores'])}", fontsize=8)
		ax.axis("off")
	fig.tight_layout()
	return fig


def plot_token_maps(model, dataset, index: int, device: str):
	"""Residual energy E, gate S, robustness R, motion prior, P_robust and the heatmap h."""
	s = dataset[index]
	p = predict(model, s, device)
	out = p["out"]
	h, w = out["enc"]["hw"]
	maps = {"E = ||r||": out["enc"]["E"], "R (robustness)": out["enc"]["R"], "S (gate)": out["enc"]["S"],
			"P_motion": out["P_motion"], "P_robust": out["P_robust"], "heatmap h": out["heatmap"].sigmoid().flatten(1)}
	fig, axes = plt.subplots(2, 4, figsize=(18, 6))
	axes[0, 0].imshow(denormalize(s["clean"])); axes[0, 0].set_title("input")
	axes[1, 0].imshow(s["motion"][0], cmap="magma"); axes[1, 0].set_title(f"raw motion (active={s['has_motion']})")
	for ax, (name, t) in zip(list(axes.flat[1:4]) + list(axes.flat[5:8]), maps.items()):
		im = ax.imshow(t[0].float().cpu().reshape(h, w), cmap="viridis")
		ax.set_title(name)
		fig.colorbar(im, ax=ax, fraction=0.046)
	for ax in axes.flat:
		ax.axis("off")
	fig.tight_layout()
	return fig


def plot_history(history: List[Dict[str, Any]]):
	ep = [r["epoch"] for r in history]
	fig, ax = plt.subplots(1, 3, figsize=(16, 4))
	ax[0].plot(ep, [r["train_loss"] for r in history]); ax[0].set_title("train loss")
	for k in ("train_sem", "train_mem", "train_rob", "train_gate", "train_pres"):
		if k in history[0]:
			ax[1].plot(ep, [r[k] for r in history], label=k[6:])
	ax[1].legend(); ax[1].set_title("PatchCvT terms")
	ax[2].plot(ep, [r["val_clean_ap50"] for r in history], label="val clean")
	ax[2].plot(ep, [r["val_degraded_ap50"] for r in history], label="val degraded")
	ax[2].legend(); ax[2].set_title("validation AP50")
	fig.tight_layout()
	return fig
