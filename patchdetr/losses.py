"""Detection (DETR) and PatchCvT objectives."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from .config import LossConfig
from .memory import object_token_mask


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
	cx, cy, w, h = b.unbind(-1)
	return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


@torch.no_grad()
def hungarian_match(logits, boxes, targets, w_cls=1.0, w_l1=5.0, w_giou=2.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
	prob = logits.float().softmax(-1)
	res = []
	for b, t in enumerate(targets):
		if len(t["labels"]) == 0:
			res.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
			continue
		tb = t["boxes"].to(boxes.device)
		pb = boxes[b].float()
		cost = (-w_cls * prob[b][:, t["labels"].to(boxes.device)]
				+ w_l1 * torch.cdist(pb, tb, p=1)
				- w_giou * generalized_box_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(tb)))
		i, j = linear_sum_assignment(cost.cpu().numpy())
		res.append((torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)))
	return res


def set_loss(logits, boxes, targets, eos_coef: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	device = logits.device
	idx = hungarian_match(logits, boxes, targets)
	tgt_cls = torch.ones(logits.shape[:2], dtype=torch.long, device=device)  # 1 = no-object
	pb, tb = [], []
	for b, (i, j) in enumerate(idx):
		i = i.to(device)
		tgt_cls[b, i] = targets[b]["labels"][j].to(device)
		pb.append(boxes[b][i])
		tb.append(targets[b]["boxes"][j].to(device))
	weight = torch.tensor([1.0, eos_coef], device=device)
	l_cls = F.cross_entropy(logits.reshape(-1, 2).float(), tgt_cls.reshape(-1), weight)
	n = max(sum(len(t["labels"]) for t in targets), 1)
	if sum(len(p) for p in pb):
		pb, tb = torch.cat(pb).float(), torch.cat(tb)
		l_l1 = F.l1_loss(pb, tb, reduction="sum") / n
		l_giou = (1 - torch.diag(generalized_box_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(tb)))).sum() / n
	else:
		l_l1 = l_giou = boxes.sum() * 0.0
	return l_cls, l_l1, l_giou


def heatmap_targets(targets, h: int, w: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Gaussian centre heatmap [B,h,w] and the exact-peak (positive) mask. The peak is the grid
	cell that contains the box centre, so every object has a positive cell."""
	ys, xs = torch.meshgrid(torch.arange(h, device=device).float(), torch.arange(w, device=device).float(), indexing="ij")
	gt = torch.zeros(len(targets), h, w, device=device)
	pos = torch.zeros_like(gt, dtype=torch.bool)
	for b, t in enumerate(targets):
		for cx, cy, bw, bh in t["boxes"].tolist():
			ix, iy = min(int(cx * w), w - 1), min(int(cy * h), h - 1)
			sx, sy = max(bw * w / 3, 0.6), max(bh * h / 3, 0.6)
			g = torch.exp(-(((xs - ix) ** 2) / (2 * sx ** 2) + ((ys - iy) ** 2) / (2 * sy ** 2)))
			gt[b] = torch.maximum(gt[b], g)
			pos[b, iy, ix] = True
	return gt, pos


def heatmap_loss(heat_logits: torch.Tensor, targets) -> torch.Tensor:
	"""CenterNet penalty-reduced focal loss for the uncertainty-aware heatmap h."""
	_, h, w = heat_logits.shape
	gt, pos = heatmap_targets(targets, h, w, heat_logits.device)
	p = heat_logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
	posf = pos.float()
	pos_l = -((1 - p) ** 2) * p.log() * posf
	neg_l = -((1 - gt) ** 4) * (p ** 2) * (1 - p).log() * (1 - posf)
	return (pos_l.sum() + neg_l.sum()) / max(int(pos.sum()), 1)


def detection_loss(out: Dict[str, Any], targets, cfg: LossConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
	total, parts = 0.0, {}
	layers = out.get("aux", []) + [out]
	for k, o in enumerate(layers):
		c, l1, g = set_loss(o["logits"], o["boxes"], targets, cfg.eos_coef)
		total = total + cfg.w_cls * c + cfg.w_l1 * l1 + cfg.w_giou * g
		if k == len(layers) - 1:
			parts.update(cls=c.detach(), l1=l1.detach(), giou=g.detach())
	hm = heatmap_loss(out["heatmap"], targets)
	parts["heat"] = hm.detach()
	return total + cfg.w_heat * hm, parts


# ---- PatchCvT objective ------------------------------------------------------------------

def semantic_loss(z_clean: torch.Tensor, z_deg: torch.Tensor, detach_clean: bool = False) -> torch.Tensor:
	"""Eq. 16: 1 - cos(z^(x), z^(x+)), on spatially aligned tokens."""
	if detach_clean:
		z_clean = z_clean.detach()
	return (1 - F.cosine_similarity(z_clean.float(), z_deg.float(), dim=-1)).mean()


def patchcvt_terms(enc: Dict[str, Any]) -> Dict[str, torch.Tensor]:
	"""L_memory (17), L_robust (19), L_gate (21). Targets exp(-||r||) are constants (detached)."""
	E, R, S = enc["E"].float(), enc["R"].float(), enc["S"].float()
	target = torch.exp(-E).detach()
	return {"memory": E.mean(), "robust": ((R - target) ** 2).mean(), "gate": (S - target).abs().mean()}


def gate_preserve_loss(S: torch.Tensor, targets, h: int, w: int) -> torch.Tensor:
	"""Deviation (default weight 0.25): object tokens should stay open, mean_{j in obj}(1 - S_j)."""
	vals = []
	for b, t in enumerate(targets):
		if len(t["boxes"]) == 0:
			continue
		m = object_token_mask(t["boxes"].numpy(), h, w).to(S.device)
		vals.append((1 - S[b][m].float()))
	return torch.cat(vals).mean() if vals else S.sum() * 0.0


class PatchDETRCriterion(nn.Module):
	"""L = 0.5 (L_det(clean) + L_det(degraded)) + w_pcvt (l1 Lsem + l2 Lmem + l3 Lrob + l4 Lgate [+ gp Lpres])."""

	def __init__(self, cfg: LossConfig) -> None:
		super().__init__()
		self.cfg = cfg

	def forward(self, out_c, out_d, targets) -> Tuple[torch.Tensor, Dict[str, float]]:
		c = self.cfg
		det_c, parts = detection_loss(out_c, targets, c)
		det_d, parts_d = detection_loss(out_d, targets, c)
		enc_c, enc_d = out_c["enc"], out_d["enc"]
		l_sem = semantic_loss(enc_c["z_hat"], enc_d["z_hat"], c.semantic_detach_clean)
		tc, td = patchcvt_terms(enc_c), patchcvt_terms(enc_d)
		l_mem, l_rob, l_gate = [(tc[k] + td[k]) / 2 for k in ("memory", "robust", "gate")]
		h, w = enc_c["hw"]
		l_pres = (gate_preserve_loss(enc_c["S"], targets, h, w) + gate_preserve_loss(enc_d["S"], targets, h, w)) / 2
		l_pcvt = (c.lam_semantic * l_sem + c.lam_memory * l_mem + c.lam_robust * l_rob
				  + c.lam_gate * l_gate + c.gate_preserve * l_pres)
		total = 0.5 * (det_c + det_d) + c.w_pcvt * l_pcvt
		logs = {k: float(v) for k, v in parts.items()}
		logs.update(det_deg=float(det_d.detach()), sem=float(l_sem.detach()), mem=float(l_mem.detach()), rob=float(l_rob.detach()),
					gate=float(l_gate.detach()), pres=float(l_pres.detach()), loss=float(total.detach()))
		return total, logs
