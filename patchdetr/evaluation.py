"""Evaluation: AP, detection metrics and prior/gate diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou

from .config import PatchDETRConfig
from .losses import cxcywh_to_xyxy, detection_loss
from .memory import object_token_mask


def average_precision(preds: Sequence[Tuple[torch.Tensor, torch.Tensor]], gts: Sequence[torch.Tensor],
					  iou_thr: float = 0.5) -> float:
	"""VOC all-point AP for one class. preds[i] = (scores [Q], xyxy [Q,4]); gts[i] = xyxy [G,4]."""
	records, n_gt = [], sum(len(g) for g in gts)
	for (s, b), g in zip(preds, gts):
		used = set()
		ious = box_iou(b, g) if len(g) else None
		for qi in s.argsort(descending=True).tolist():
			tp = 0
			if ious is not None:
				cand = [(float(ious[qi, j]), j) for j in range(len(g)) if j not in used]
				if cand:
					best, j = max(cand)
					if best >= iou_thr:
						used.add(j)
						tp = 1
			records.append((float(s[qi]), tp))
	if n_gt == 0 or not records:
		return 0.0
	records.sort(key=lambda r: -r[0])
	tp = np.cumsum([r[1] for r in records])
	fp = np.cumsum([1 - r[1] for r in records])
	rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
	mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
	for i in range(len(mpre) - 2, -1, -1):
		mpre[i] = max(mpre[i], mpre[i + 1])
	idx = np.where(mrec[1:] != mrec[:-1])[0]
	return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def recall_at(preds, gts, score_thr: float = 0.5, iou_thr: float = 0.5) -> float:
	hit = total = 0
	for (s, b), g in zip(preds, gts):
		total += len(g)
		if len(g) == 0:
			continue
		keep = s >= score_thr
		if keep.any():
			ious = box_iou(g, b[keep])
			hit += int((ious.max(1).values >= iou_thr).sum())
	return hit / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, cfg: PatchDETRConfig, device: str, prefix: str = "") -> Dict[str, float]:
	"""Returns {prefix+val_loss, prefix+ap50, prefix+recall50}. Never updates the model."""
	model.eval()
	amp = cfg.train.amp and device == "cuda"
	preds, gts, tot, n = [], [], 0.0, 0
	for batch in loader:
		clean, motion = batch["clean"].to(device), batch["motion"].to(device)
		with torch.autocast(device, dtype=torch.float16, enabled=amp):
			out = model(clean, motion)
		loss, _ = detection_loss({**out, "aux": []}, batch["targets"], cfg.loss)
		tot += float(loss) * len(clean)
		n += len(clean)
		scores = 1 - out["logits"].float().softmax(-1)[..., 1]
		for b, t in enumerate(batch["targets"]):
			preds.append((scores[b].cpu(), cxcywh_to_xyxy(out["boxes"][b].float()).cpu()))
			gts.append(cxcywh_to_xyxy(t["boxes"]))
	return {prefix + "loss": tot / max(n, 1), prefix + "ap50": average_precision(preds, gts, 0.5),
			prefix + "recall50": recall_at(preds, gts)}


@torch.no_grad()
def prior_diagnostics(model, loader, device: str, max_batches: int = 20) -> Dict[str, float]:
	"""Mean residual energy E, gate S and priors inside labelled objects vs. background, plus the
	fraction of samples whose temporal prior is active."""
	model.eval()
	acc: Dict[str, List[float]] = {k: [] for k in ("E_obj", "E_bg", "S_obj", "S_bg", "Pm_obj", "Pm_bg", "Pr_obj", "Pr_bg", "motion_active")}
	for bi, batch in enumerate(loader):
		if bi >= max_batches:
			break
		out = model(batch["clean"].to(device), batch["motion"].to(device))
		h, w = out["enc"]["hw"]
		acc["motion_active"] += batch["has_motion"].float().tolist()
		for b, t in enumerate(batch["targets"]):
			if len(t["boxes"]) == 0:
				continue
			m = object_token_mask(t["boxes"].numpy(), h, w).to(device)
			for key, tensor in (("E", out["enc"]["E"]), ("S", out["enc"]["S"]), ("Pm", out["P_motion"]), ("Pr", out["P_robust"])):
				acc[key + "_obj"].append(float(tensor[b][m].mean()))
				acc[key + "_bg"].append(float(tensor[b][~m].mean()))
	return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
