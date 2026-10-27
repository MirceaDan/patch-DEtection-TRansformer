"""PatchDETR (PatchCvDetr) training.

Implements patch-cv-detr.pdf (sec. 11-12) on top of the PatchCvT formulation
(patchCvT/PatchCvT_Extended_Formulation.pdf, patchCvT/train.py):

  F = f_theta(x)                          shared (frozen) ResNet18 backbone      (31)
  q_j        = L2-normalised patch of F                                           (4)
  q~_j       = sum_k softmax_k(-||q_j - m_k||/tau) m_k   fixed memory bank        (21-23)
  r_j        = q_j - q~_j ; E_j = ||r_j|| ; a_j = ||F_j||                         (24-25, 10)
  z_j        = [q_j || q~_j || r_j || a_j]                                        (26, 11)
  z'_j       = Transformer(z_j) ; R_j = g_phi(z'_j)                               (27-28)
  S_j        = sigmoid(W_r R_j - W_e ||r_j||) ; z^_j = S_j z'_j                   (29-30)
  P_robust   = sigmoid(gamma (E - delta)), P_motion = frame-difference map        (32-33)
  A          = softmax(QK^T/sqrt(d) + l1 P_motion + l2 P_robust)  (decoder x-attn)(34)
  Y^         = DETR(Z^)                                                           (35)

Loss = DETR loss (Hungarian: CE + L1 + GIoU, aux layers, uncertainty heatmap)
     + PatchCvT loss  l1 L_semantic + l2 L_memory + l3 L_robust + l4 L_gate  (eq. 16-22)

Layout expected next to this file: database/{train,test}/*.jpg|txt (YOLO labels)
and memorybank/*.jpg|txt (normal samples, used to build the fixed memory bank).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import generalized_box_iou, box_iou

ROOT = os.path.dirname(os.path.abspath(__file__))
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# PatchCvT objective weights (eq. 23-26 of PatchCvT_Extended_Formulation.pdf)
PCVT_LAMBDAS = {"semantic": 1.0, "memory": 0.5, "robust": 0.25, "gate": 0.25}


################################################################################
# DATA
################################################################################

def read_yolo(label_path: str) -> np.ndarray:
	"""Returns [n,5] array: class, cx, cy, w, h (normalised)."""
	if not os.path.isfile(label_path):
		return np.zeros((0, 5), dtype=np.float32)
	rows = []
	with open(label_path, encoding="utf-8") as f:
		for ln, line in enumerate(f, 1):
			v = line.split()
			if not v:
				continue
			if len(v) < 5:
				raise ValueError(f"Bad YOLO line {label_path}:{ln}")
			rows.append([float(x) for x in v[:5]])
	arr = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
	return arr[(arr[:, 3] > 0) & (arr[:, 4] > 0)]


def list_images(folder: str) -> List[str]:
	if not os.path.isdir(folder):
		raise FileNotFoundError(folder)
	return sorted(
		os.path.join(d, n)
		for d, _, names in os.walk(folder)
		for n in names
		if os.path.splitext(n)[1].lower() in VALID_EXT
	)


_FRAME_RE = re.compile(r"^(.+_)(\d{6})_(\d{2}-\d{2}-\d{2})\.\w+$")


def build_frame_index(paths: Sequence[str]) -> Dict[Tuple[str, str, int], str]:
	index = {}
	for p in paths:
		m = _FRAME_RE.match(os.path.basename(p))
		if m:
			index[(os.path.dirname(p), m.group(1), int(m.group(2)))] = p
	return index


def load_image(path: str, size: Tuple[int, int]) -> Image.Image:
	"""Loads and resizes to (W,H); JPEG draft mode makes 4K frames cheap to decode."""
	with Image.open(path) as im:
		im.draft("RGB", (size[0] * 2, size[1] * 2))
		return im.convert("RGB").resize(size, Image.BILINEAR)


class MudDegradation:
	"""Simulates the target failure mode: blur / JPEG / lighting / dirt, and mud
	that partially or fully covers the labelled objects. Geometry is preserved so
	boxes stay valid and clean/degraded tokens stay spatially aligned."""

	def __init__(self, mud_prob: float = 0.7) -> None:
		self.mud_prob = mud_prob

	@staticmethod
	def _mud_patch(box: Tuple[int, int, int, int], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
		x0, y0, x1, y1 = box
		h, w = max(1, y1 - y0), max(1, x1 - x0)
		coverage = random.uniform(0.3, 1.0)
		yy, xx = np.mgrid[0:h, 0:w]
		cx, cy = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
		dist = ((xx - cx) / (w * coverage)) ** 2 + ((yy - cy) / (h * coverage)) ** 2
		noise = np.random.rand(max(1, h // 3 + 1), max(1, w // 3 + 1))
		noise = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255.0
		alpha = np.clip((1.2 - dist) * 2.0 + (noise - 0.5), 0, 1) * random.uniform(0.7, 1.0)
		base = np.array([random.randint(70, 120), random.randint(50, 90), random.randint(30, 60)], dtype=np.float32)
		color = base[None, None, :] * (0.7 + 0.6 * noise[..., None])
		return alpha[..., None].astype(np.float32), color.astype(np.float32)

	def __call__(self, image: Image.Image, boxes_xyxy: np.ndarray) -> Image.Image:
		if random.random() < 0.5:
			image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.5)))
		if random.random() < 0.7:
			image = TF.adjust_brightness(image, random.uniform(0.5, 1.4))
			image = TF.adjust_contrast(image, random.uniform(0.5, 1.4))
			image = TF.adjust_saturation(image, random.uniform(0.6, 1.2))
		arr = np.asarray(image).astype(np.float32)
		H, W = arr.shape[:2]
		for b in boxes_xyxy:
			if random.random() > self.mud_prob:
				continue
			x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
			x1, y1 = min(W, int(math.ceil(b[2]))), min(H, int(math.ceil(b[3])))
			if x1 - x0 < 2 or y1 - y0 < 2:
				continue
			alpha, color = self._mud_patch((x0, y0, x1, y1), (H, W))
			arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1 - alpha) + color * alpha
		image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
		if random.random() < 0.6:  # JPEG compression artefacts
			import io
			buf = io.BytesIO()
			image.save(buf, "JPEG", quality=random.randint(15, 60))
			buf.seek(0)
			image = Image.open(buf).convert("RGB")
		return image


class DetectionDataset(Dataset):
	def __init__(self, folder: str, size: Tuple[int, int], train: bool, limit: Optional[int] = None) -> None:
		self.paths = list_images(folder)
		if limit:
			self.paths = self.paths[:limit]
		self.size = size  # (W,H)
		self.train = train
		self.frame_index = build_frame_index(list_images(folder))
		self.degrade = MudDegradation()

	def __len__(self) -> int:
		return len(self.paths)

	def motion_prior(self, path: str, image: Image.Image) -> np.ndarray:
		"""P_motion (eq. 32): |frame_t - frame_{t-1}| from the preceding frame of the same
		sequence; zeros when there is no previous frame (isolated images)."""
		W, H = image.size
		m = _FRAME_RE.match(os.path.basename(path))
		prev = self.frame_index.get((os.path.dirname(path), m.group(1), int(m.group(2)) - 1)) if m else None
		if prev is None:
			return np.zeros((H, W), dtype=np.float32)
		a = np.asarray(image.convert("L").resize((W // 4, H // 4)), dtype=np.float32)
		b = np.asarray(load_image(prev, (W, H)).convert("L").resize((W // 4, H // 4)), dtype=np.float32)
		d = np.abs(a - b)
		d = np.asarray(Image.fromarray(d.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
		d = d / max(float(np.percentile(d, 99.5)), 8.0)
		return np.clip(d, 0, 1).astype(np.float32)

	def __getitem__(self, i: int) -> Dict[str, Any]:
		path = self.paths[i]
		image = load_image(path, self.size)
		lab = read_yolo(os.path.splitext(path)[0] + ".txt")
		motion = self.motion_prior(path, image)
		boxes = lab[:, 1:5].copy()  # cx cy w h
		if self.train and random.random() < 0.5:
			image = image.transpose(Image.FLIP_LEFT_RIGHT)
			motion = motion[:, ::-1].copy()
			boxes[:, 0] = 1 - boxes[:, 0]
		W, H = self.size
		xyxy = np.stack(
			[(boxes[:, 0] - boxes[:, 2] / 2) * W, (boxes[:, 1] - boxes[:, 3] / 2) * H,
			 (boxes[:, 0] + boxes[:, 2] / 2) * W, (boxes[:, 1] + boxes[:, 3] / 2) * H], 1
		) if len(boxes) else np.zeros((0, 4), np.float32)
		degraded = self.degrade(image, xyxy) if self.train else None

		def to_t(im: Image.Image) -> torch.Tensor:
			return TF.normalize(TF.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD)

		return {
			"clean": to_t(image),
			"degraded": to_t(degraded) if degraded is not None else None,
			"motion": torch.from_numpy(motion)[None],
			"boxes": torch.from_numpy(boxes),
			"labels": torch.zeros(len(boxes), dtype=torch.long),  # single class: TCD
		}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
	out: Dict[str, Any] = {
		"clean": torch.stack([b["clean"] for b in batch]),
		"motion": torch.stack([b["motion"] for b in batch]),
		"targets": [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch],
	}
	out["degraded"] = torch.stack([b["degraded"] for b in batch]) if batch[0]["degraded"] is not None else None
	return out


################################################################################
# PATCHCVT (adapted from patchCvT/train.py: frozen ResNet18 + token transformer)
################################################################################

class ResNetBackbone(nn.Module):
	"""f_theta: frozen ImageNet ResNet18 (stem, layer1, layer2[, layer3]) -> stride 8/16 map."""

	def __init__(self, pretrained: bool, use_layer3: bool) -> None:
		super().__init__()
		try:
			net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
		except Exception as exc:  # offline fallback
			print(f"[WARN] could not load pretrained ResNet18 ({exc}); using random init")
			net = resnet18(weights=None)
		layers = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2]
		self.out_channels = 128
		if use_layer3:
			layers.append(net.layer3)
			self.out_channels = 256
		self.body = nn.Sequential(*layers)
		for p in self.parameters():
			p.requires_grad = False
		self.eval()

	def train(self, mode: bool = True) -> "ResNetBackbone":
		return super().train(False)  # BN statistics stay frozen

	@torch.no_grad()
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.body(x)


def sine_position_embedding(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
	"""2D sine positional encoding -> [1, h*w, dim]."""
	d = dim // 4
	omega = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=torch.float32) / d))
	ys, xs = torch.meshgrid(torch.arange(h, device=device).float(), torch.arange(w, device=device).float(), indexing="ij")
	oy, ox = ys.reshape(-1, 1) * omega, xs.reshape(-1, 1) * omega
	return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)[None]


class PatchCvTModule(nn.Module):
	"""Memory-conditioned tokenisation + transformer + robustness head + gate."""

	def __init__(self, in_channels: int, dim: int, depth: int, heads: int, tau: float, dropout: float) -> None:
		super().__init__()
		self.dim, self.tau = dim, tau
		self.register_buffer("memory", torch.zeros(0, in_channels))  # M, fixed (eq. 18)
		self.proj = nn.Linear(3 * in_channels + 1, dim)  # z_j = [q||q~||r||a]
		layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
		self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
		self.robust_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))  # g_phi
		self.w_r = nn.Parameter(torch.tensor(4.0))
		self.w_e = nn.Parameter(torch.tensor(1.0))

	def set_memory(self, memory: torch.Tensor) -> None:
		self.memory = memory.to(self.memory.device, torch.float32)

	def retrieve(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Soft retrieval (eq. 21-23); returns q~ and r. Memory never receives gradients."""
		M = self.memory
		d = torch.cdist(q.float(), M.unsqueeze(0).expand(q.shape[0], -1, -1))  # [B,N,K]
		alpha = torch.softmax(-d / self.tau, dim=-1)
		q_tilde = alpha @ M
		return q_tilde, q - q_tilde

	def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
		B, C, h, w = feat.shape
		F_tok = feat.flatten(2).transpose(1, 2)  # [B,N,C]
		a = F_tok.norm(dim=-1, keepdim=True)  # activation energy (eq. 10)
		q = F.normalize(F_tok, dim=-1)  # q_j
		with torch.no_grad():
			q_tilde, _ = self.retrieve(q)
		r = q - q_tilde
		r_norm = r.norm(dim=-1)  # E_j = ||r_j|| (eq. 25)
		z = torch.cat([q, q_tilde, r, a], dim=-1)  # eq. 26
		x = self.proj(z) + sine_position_embedding(h, w, self.dim, feat.device)
		z_prime = self.encoder(x)  # eq. 27
		R = torch.sigmoid(self.robust_head(z_prime)).squeeze(-1)  # eq. 28, in (0,1)
		S = torch.sigmoid(self.w_r * R - self.w_e * r_norm)  # eq. 29
		z_hat = S.unsqueeze(-1) * z_prime  # eq. 30
		return {"z_hat": z_hat, "R": R, "S": S, "E": r_norm, "r_norm": r_norm, "hw": (h, w)}


@torch.no_grad()
def build_memory_bank(backbone: ResNetBackbone, folder: str, size: Tuple[int, int], memory_size: int,
					  device: torch.device, per_image_bg: int = 60, per_image_obj: int = 40,
					  limit: Optional[int] = None) -> torch.Tensor:
	"""M = {m_k} (eq. 5): L2-normalised backbone patches of normal images (labelled objects +
	background), reduced to K items with PatchCore-style greedy k-center coreset."""
	paths = list_images(folder)[:limit] if limit else list_images(folder)
	if not paths:
		raise RuntimeError(f"No images in memory bank folder {folder}")
	rng = np.random.default_rng(0)
	pool = []
	for i, p in enumerate(paths):
		x = TF.normalize(TF.to_tensor(load_image(p, size)), IMAGENET_MEAN, IMAGENET_STD)[None].to(device)
		fm = backbone(x)
		_, C, h, w = fm.shape
		tok = F.normalize(fm.flatten(2).transpose(1, 2)[0], dim=-1)
		lab = read_yolo(os.path.splitext(p)[0] + ".txt")
		inside = torch.zeros(h * w, dtype=torch.bool)
		for _, cx, cy, bw, bh in lab:
			xs0, xs1 = int(max(0, (cx - bw / 2) * w)), int(math.ceil(min(1, cx + bw / 2) * w))
			ys0, ys1 = int(max(0, (cy - bh / 2) * h)), int(math.ceil(min(1, cy + bh / 2) * h))
			m = torch.zeros(h, w, dtype=torch.bool)
			m[ys0:max(ys1, ys0 + 1), xs0:max(xs1, xs0 + 1)] = True
			inside |= m.flatten()
		obj_idx, bg_idx = inside.nonzero().flatten().numpy(), (~inside).nonzero().flatten().numpy()
		sel = np.concatenate([
			rng.choice(obj_idx, min(per_image_obj, len(obj_idx)), replace=False) if len(obj_idx) else [],
			rng.choice(bg_idx, min(per_image_bg, len(bg_idx)), replace=False),
		]).astype(np.int64)
		pool.append(tok[torch.from_numpy(sel).to(device)].cpu())
		if (i + 1) % 100 == 0:
			print(f"[MEMORY] {i + 1}/{len(paths)} images")
	pool_t = torch.cat(pool).to(device)
	k = min(memory_size, pool_t.shape[0])
	# greedy k-center coreset
	chosen = [int(rng.integers(pool_t.shape[0]))]
	min_d = torch.cdist(pool_t, pool_t[chosen[0]][None]).squeeze(1)
	for _ in range(k - 1):
		nxt = int(min_d.argmax())
		chosen.append(nxt)
		min_d = torch.minimum(min_d, torch.cdist(pool_t, pool_t[nxt][None]).squeeze(1))
	bank = pool_t[chosen].cpu()
	print(f"[MEMORY] pool={pool_t.shape[0]} tokens -> bank {tuple(bank.shape)} from {len(paths)} images")
	return bank


################################################################################
# DETR WITH ATTENTION MODULATION
################################################################################

class MLP(nn.Module):
	def __init__(self, i: int, h: int, o: int, n: int) -> None:
		super().__init__()
		dims = [i] + [h] * (n - 1) + [o]
		self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for i, l in enumerate(self.layers):
			x = F.relu(l(x)) if i < len(self.layers) - 1 else l(x)
		return x


class ModulatedDecoderLayer(nn.Module):
	"""DETR decoder layer whose cross-attention logits get the additive bias of eq. 34:
	softmax(QK^T/sqrt(d) + l1 P_motion + l2 P_robust)."""

	def __init__(self, dim: int, heads: int, dropout: float) -> None:
		super().__init__()
		self.heads = heads
		self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
		self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
		self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
		self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
		self.drop = nn.Dropout(dropout)

	def forward(self, tgt, query_pos, memory, mem_pos, bias):
		q = k = tgt + query_pos
		tgt = self.n1(tgt + self.drop(self.self_attn(q, k, tgt, need_weights=False)[0]))
		mask = bias[:, None, None, :].expand(-1, self.heads, tgt.shape[1], -1).reshape(-1, tgt.shape[1], bias.shape[-1])
		attn = self.cross_attn(tgt + query_pos, memory + mem_pos, memory, attn_mask=mask.to(tgt.dtype), need_weights=False)[0]
		tgt = self.n2(tgt + self.drop(attn))
		return self.n3(tgt + self.drop(self.ffn(tgt)))


class PatchDETR(nn.Module):
	def __init__(self, a: argparse.Namespace) -> None:
		super().__init__()
		D = a.dim
		self.backbone = ResNetBackbone(not a.no_pretrained, not a.stride8)
		self.patchcvt = PatchCvTModule(self.backbone.out_channels, D, a.enc_layers, a.heads, a.tau, a.dropout)
		self.queries = nn.Embedding(a.num_queries, D)
		self.query_pos = nn.Embedding(a.num_queries, D)
		self.decoder = nn.ModuleList(ModulatedDecoderLayer(D, a.heads, a.dropout) for _ in range(a.dec_layers))
		self.dec_norm = nn.LayerNorm(D)
		self.class_head = nn.Linear(D, 2)  # TCD + no-object
		self.box_head = MLP(D, D, 4, 3)  # (x, y, w, h)
		self.heat_head = nn.Linear(D, 1)  # uncertainty-aware heatmap h
		# attention-modulation parameters (eq. 33-34)
		self.lam_motion = nn.Parameter(torch.tensor(1.0))
		self.lam_robust = nn.Parameter(torch.tensor(1.0))
		self.gamma = nn.Parameter(torch.tensor(2.0))
		self.delta = nn.Parameter(torch.tensor(1.0))

	def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
		return self.patchcvt(self.backbone(images))

	def detect(self, enc: Dict[str, torch.Tensor], motion: torch.Tensor) -> Dict[str, Any]:
		z_hat, (h, w) = enc["z_hat"], enc["hw"]
		B, N, D = z_hat.shape
		P_motion = F.adaptive_avg_pool2d(motion, (h, w)).flatten(1)  # eq. 32
		E = enc["E"]
		E_rel = E / (E.mean(dim=1, keepdim=True).detach() + 1e-6)  # scale-free energy
		P_robust = torch.sigmoid(self.gamma * (E_rel - self.delta))  # eq. 33
		bias = self.lam_motion * P_motion + self.lam_robust * P_robust  # eq. 34 additive term
		mem_pos = sine_position_embedding(h, w, D, z_hat.device)
		qpos = self.query_pos.weight[None].expand(B, -1, -1)
		tgt = self.queries.weight[None].expand(B, -1, -1)
		outs = []
		for layer in self.decoder:
			tgt = layer(tgt, qpos, z_hat, mem_pos, bias)
			hs = self.dec_norm(tgt)
			outs.append({"logits": self.class_head(hs), "boxes": self.box_head(hs).sigmoid()})
		heat = self.heat_head(z_hat).squeeze(-1).reshape(B, h, w)
		return {"logits": outs[-1]["logits"], "boxes": outs[-1]["boxes"], "aux": outs[:-1], "heatmap": heat}


################################################################################
# LOSSES
################################################################################

def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
	cx, cy, w, h = b.unbind(-1)
	return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


@torch.no_grad()
def hungarian_match(logits, boxes, targets, w_cls=1.0, w_l1=5.0, w_giou=2.0):
	prob = logits.softmax(-1)
	res = []
	for b, t in enumerate(targets):
		if len(t["labels"]) == 0:
			res.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
			continue
		tb = t["boxes"].to(boxes.device)
		cost = (-w_cls * prob[b][:, t["labels"]]
				+ w_l1 * torch.cdist(boxes[b], tb, p=1)
				- w_giou * generalized_box_iou(cxcywh_to_xyxy(boxes[b]), cxcywh_to_xyxy(tb)))
		i, j = linear_sum_assignment(cost.cpu().numpy())
		res.append((torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)))
	return res


def set_loss(logits, boxes, targets, eos_coef=0.1):
	device = logits.device
	idx = hungarian_match(logits.float(), boxes.float(), targets)
	tgt_cls = torch.ones(logits.shape[:2], dtype=torch.long, device=device)  # 1 = no-object
	pb, tb = [], []
	for b, (i, j) in enumerate(idx):
		tgt_cls[b, i.to(device)] = targets[b]["labels"][j].to(device)
		pb.append(boxes[b][i.to(device)])
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


def heatmap_loss(heat_logits: torch.Tensor, targets) -> torch.Tensor:
	"""CenterNet-style focal loss for the uncertainty-aware heatmap h."""
	B, h, w = heat_logits.shape
	ys, xs = torch.meshgrid(torch.arange(h, device=heat_logits.device), torch.arange(w, device=heat_logits.device), indexing="ij")
	gt = torch.zeros(B, h, w, device=heat_logits.device)
	npos = 0
	for b, t in enumerate(targets):
		for cx, cy, bw, bh in t["boxes"].tolist():
			sx, sy = max(bw * w / 3, 0.6), max(bh * h / 3, 0.6)
			g = torch.exp(-(((xs - cx * w + 0.5) ** 2) / (2 * sx ** 2) + ((ys - cy * h + 0.5) ** 2) / (2 * sy ** 2)))
			gt[b] = torch.maximum(gt[b], g)
			npos += 1
	p = heat_logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
	pos = gt.ge(0.99).float()
	pos_l = -((1 - p) ** 2) * p.log() * pos
	neg_l = -((1 - gt) ** 4) * (p ** 2) * (1 - p).log() * (1 - pos)
	return (pos_l.sum() + neg_l.sum()) / max(npos, 1)


def detection_loss(out: Dict[str, Any], targets, a) -> Tuple[torch.Tensor, Dict[str, float]]:
	total, parts = 0.0, {}
	for k, o in enumerate(out["aux"] + [out]):
		c, l1, g = set_loss(o["logits"], o["boxes"], targets)
		total = total + a.w_cls * c + a.w_l1 * l1 + a.w_giou * g
		if k == len(out["aux"]):
			parts.update(cls=float(c.detach()), l1=float(l1.detach()), giou=float(g.detach()))
	hm = heatmap_loss(out["heatmap"], targets)
	parts["heat"] = float(hm)
	return total + a.w_heat * hm, parts


def patchcvt_regularizers(enc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""L_memory (17), L_robust (19), L_gate (21). Targets exp(-||r||) are constants."""
	r = enc["r_norm"]
	target = torch.exp(-r).detach()
	return r.mean(), ((enc["R"] - target) ** 2).mean(), (enc["S"] - target).abs().mean()


def semantic_loss(z_clean: torch.Tensor, z_deg: torch.Tensor) -> torch.Tensor:
	"""L_semantic (16): 1 - cos(z^(x), z^(x+)), averaged over spatially aligned tokens."""
	return (1 - F.cosine_similarity(z_clean, z_deg, dim=-1)).mean()


################################################################################
# TRAIN / EVAL
################################################################################

def train_step(model, batch, a, device, amp) -> Tuple[torch.Tensor, Dict[str, float]]:
	clean = batch["clean"].to(device, non_blocking=True)
	degraded = batch["degraded"].to(device, non_blocking=True)
	motion = batch["motion"].to(device, non_blocking=True)
	targets = batch["targets"]
	with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
		enc_c, enc_d = model.encode(clean), model.encode(degraded)
		det_c, det_d = model.detect(enc_c, motion), model.detect(enc_d, motion)
	l_det_c, parts = detection_loss(det_c, targets, a)
	l_det_d, _ = detection_loss(det_d, targets, a)
	l_sem = semantic_loss(enc_c["z_hat"].float(), enc_d["z_hat"].float())
	reg_c, reg_d = patchcvt_regularizers(enc_c), patchcvt_regularizers(enc_d)
	l_mem, l_rob, l_gate = [(x + y) / 2 for x, y in zip(reg_c, reg_d)]
	l_pcvt = (PCVT_LAMBDAS["semantic"] * l_sem + PCVT_LAMBDAS["memory"] * l_mem
			  + PCVT_LAMBDAS["robust"] * l_rob + PCVT_LAMBDAS["gate"] * l_gate)
	loss = 0.5 * (l_det_c + l_det_d) + a.w_pcvt * l_pcvt
	parts.update(det_deg=float(l_det_d), sem=float(l_sem), mem=float(l_mem), rob=float(l_rob), gate=float(l_gate))
	return loss, parts


@torch.no_grad()
def evaluate(model, loader, a, device, amp) -> Dict[str, float]:
	model.eval()
	preds, gts, tot, n = [], [], 0.0, 0
	for batch in loader:
		clean, motion = batch["clean"].to(device), batch["motion"].to(device)
		with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
			det = model.detect(model.encode(clean), motion)
		loss, _ = detection_loss({**det, "aux": []}, batch["targets"], a)
		tot += float(loss) * len(clean)
		n += len(clean)
		scores = det["logits"].float().softmax(-1)[..., 0]
		for b, t in enumerate(batch["targets"]):
			preds.append((scores[b].cpu(), cxcywh_to_xyxy(det["boxes"][b].float()).cpu()))
			gts.append(cxcywh_to_xyxy(t["boxes"]))
	return {"val_loss": tot / max(n, 1), "ap50": average_precision(preds, gts, 0.5)}


def average_precision(preds, gts, iou_thr: float) -> float:
	records, n_gt = [], sum(len(g) for g in gts)
	for img, ((s, b), g) in enumerate(zip(preds, gts)):
		used = set()
		order = s.argsort(descending=True)
		ious = box_iou(b, g) if len(g) else None
		for qi in order.tolist():
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


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train PatchDETR (PatchCvT-modulated DETR)")
	p.add_argument("--train-dir", default=os.path.join(ROOT, "database", "train"))
	p.add_argument("--test-dir", default=os.path.join(ROOT, "database", "test"))
	p.add_argument("--memory-dir", default=os.path.join(ROOT, "memorybank"))
	p.add_argument("--output-dir", default=os.path.join(ROOT, "model"))
	p.add_argument("--img-w", type=int, default=1024)
	p.add_argument("--img-h", type=int, default=576)
	p.add_argument("--stride8", action="store_true", help="use ResNet18 layer2 only (stride 8; many more tokens)")
	p.add_argument("--no-pretrained", action="store_true")
	p.add_argument("--memory-size", type=int, default=2048)
	p.add_argument("--rebuild-memory", action="store_true")
	p.add_argument("--tau", type=float, default=0.1)
	p.add_argument("--dim", type=int, default=256)
	p.add_argument("--heads", type=int, default=8)
	p.add_argument("--enc-layers", type=int, default=2)
	p.add_argument("--dec-layers", type=int, default=4)
	p.add_argument("--num-queries", type=int, default=50)
	p.add_argument("--dropout", type=float, default=0.1)
	p.add_argument("--epochs", type=int, default=50)
	p.add_argument("--batch-size", type=int, default=2)
	p.add_argument("--lr", type=float, default=1e-4)
	p.add_argument("--weight-decay", type=float, default=1e-4)
	p.add_argument("--clip", type=float, default=0.1)
	p.add_argument("--workers", type=int, default=4)
	p.add_argument("--w-cls", type=float, default=1.0)
	p.add_argument("--w-l1", type=float, default=5.0)
	p.add_argument("--w-giou", type=float, default=2.0)
	p.add_argument("--w-heat", type=float, default=1.0)
	p.add_argument("--w-pcvt", type=float, default=1.0, help="weight of the PatchCvT objective (eq. 22)")
	p.add_argument("--seed", type=int, default=23)
	p.add_argument("--limit", type=int, default=None, help="cap images per split (smoke tests)")
	p.add_argument("--resume", default=None)
	p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	return p.parse_args()


def main() -> None:
	a = parse_args()
	random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
	device = torch.device(a.device)
	amp = device.type == "cuda"
	os.makedirs(a.output_dir, exist_ok=True)
	size = (a.img_w, a.img_h)
	stride = 8 if a.stride8 else 16
	if a.img_w % stride or a.img_h % stride:
		raise SystemExit(f"image size must be a multiple of {stride}")

	model = PatchDETR(a).to(device)

	# Fixed memory bank (eq. 5, 18) built from the memorybank/ images
	bank_path = os.path.join(a.output_dir, "memory_bank.pt")
	if os.path.isfile(bank_path) and not a.rebuild_memory:
		bank = torch.load(bank_path, map_location="cpu")["memory_bank"]
		print(f"[MEMORY] loaded {tuple(bank.shape)} from {bank_path}")
	else:
		bank = build_memory_bank(model.backbone, a.memory_dir, size, a.memory_size, device, limit=a.limit)
		torch.save({"memory_bank": bank, "img_size": size}, bank_path)
	model.patchcvt.set_memory(bank.to(device))

	train_ds = DetectionDataset(a.train_dir, size, True, a.limit)
	test_ds = DetectionDataset(a.test_dir, size, False, a.limit)
	print(f"[DATA] train={len(train_ds)} test={len(test_ds)}")
	kw = dict(num_workers=a.workers, collate_fn=collate, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
	train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, drop_last=len(train_ds) >= a.batch_size, **kw)
	test_dl = DataLoader(test_ds, a.batch_size, shuffle=False, **kw)

	params = [p for p in model.parameters() if p.requires_grad]
	opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
	scaler = torch.amp.GradScaler(enabled=amp)
	print(f"[MODEL] trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M | device={device} amp={amp}")

	start, best = 0, -1.0
	if a.resume:
		ck = torch.load(a.resume, map_location=device)
		model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
		start, best = ck["epoch"] + 1, ck["best_ap50"]

	log_path = os.path.join(a.output_dir, "train_log.csv")
	for epoch in range(start, a.epochs):
		model.train()
		t0, agg, steps = time.time(), {}, 0
		for batch in train_dl:
			loss, parts = train_step(model, batch, a, device, amp)
			opt.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.unscale_(opt)
			nn.utils.clip_grad_norm_(params, a.clip)
			scaler.step(opt); scaler.update()
			parts["loss"] = float(loss)
			for k, v in parts.items():
				agg[k] = agg.get(k, 0.0) + v
			steps += 1
		sched.step()
		agg = {k: v / max(steps, 1) for k, v in agg.items()}
		ev = evaluate(model, test_dl, a, device, amp)
		print(f"[EPOCH {epoch + 1}/{a.epochs}] {time.time() - t0:.0f}s " + " ".join(f"{k}={v:.4f}" for k, v in {**agg, **ev}.items()))
		row = {"epoch": epoch + 1, **agg, **ev}
		new = not os.path.isfile(log_path)
		with open(log_path, "a", newline="") as f:
			w = csv.DictWriter(f, fieldnames=list(row)); 
			if new:
				w.writeheader()
			w.writerow(row)
		state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
				 "epoch": epoch, "best_ap50": max(best, ev["ap50"]), "args": vars(a)}
		torch.save(state, os.path.join(a.output_dir, "last.pt"))
		if ev["ap50"] > best:
			best = ev["ap50"]
			torch.save(state, os.path.join(a.output_dir, "best.pt"))
	print(f"[DONE] best AP50={best:.4f}")


if __name__ == "__main__":
	main()
