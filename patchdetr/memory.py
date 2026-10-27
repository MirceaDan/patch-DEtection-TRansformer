"""Fixed memory bank M (eq. 5) from the memorybank/ images."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import PatchDETRConfig
from .data import list_images, load_image, read_yolo, subsample, to_tensor
from .model import ResNetBackbone


def object_token_mask(boxes_cxcywh: np.ndarray, h: int, w: int) -> torch.Tensor:
	"""[h*w] bool mask of tokens whose cell overlaps a labelled box (>=1 cell per box)."""
	mask = torch.zeros(h, w, dtype=torch.bool)
	for cx, cy, bw, bh in boxes_cxcywh:
		x0, x1 = int(max(0, math.floor((cx - bw / 2) * w))), int(min(w, math.ceil((cx + bw / 2) * w)))
		y0, y1 = int(max(0, math.floor((cy - bh / 2) * h))), int(min(h, math.ceil((cy + bh / 2) * h)))
		mask[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True
	return mask.flatten()


def greedy_coreset(pool: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
	"""PatchCore greedy k-center selection; returns indices [k]."""
	g = torch.Generator().manual_seed(seed)
	k = min(k, pool.shape[0])
	first = int(torch.randint(pool.shape[0], (1,), generator=g))
	chosen = [first]
	min_d = torch.cdist(pool, pool[first][None]).squeeze(1)
	for _ in range(k - 1):
		nxt = int(min_d.argmax())
		chosen.append(nxt)
		min_d = torch.minimum(min_d, torch.cdist(pool, pool[nxt][None]).squeeze(1))
	return torch.tensor(chosen)


def memory_fingerprint(cfg: PatchDETRConfig, backbone: ResNetBackbone, n_images: int) -> Dict[str, Any]:
	"""Everything that makes a cached bank invalid if changed."""
	return {"img_size": [cfg.data.img_w, cfg.data.img_h], "stride8": cfg.model.stride8,
			"pretrained_loaded": backbone.pretrained_loaded, "memory_size": cfg.model.memory_size,
			"n_images": n_images, "channels": backbone.out_channels,
			"trained_tail": backbone.trainable_tail}


@torch.no_grad()
def build_memory_bank(backbone: ResNetBackbone, cfg: PatchDETRConfig, device: str,
					  per_image_obj: int = 40, per_image_bg: int = 60, verbose: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
	"""Samples L2-normalised patch tokens (labelled objects + background) of the normal images
	and reduces them to K entries with a k-center coreset. Uses the exact same resize /
	normalisation / backbone as training, so bank and query tokens live in the same space."""
	paths = subsample(list_images(cfg.data.memory_dir), cfg.data.limit)
	if not paths:
		raise RuntimeError(f"No images in {cfg.data.memory_dir}")
	rng = np.random.default_rng(0)
	size = (cfg.data.img_w, cfg.data.img_h)
	backbone.eval()
	pool = []
	for i, p in enumerate(paths):
		x = to_tensor(load_image(p, size))[None].to(device)
		fm = backbone(x)
		_, _, h, w = fm.shape
		tok = F.normalize(fm.flatten(2).transpose(1, 2)[0].float(), dim=-1).cpu()
		inside = object_token_mask(read_yolo(os.path.splitext(p)[0] + ".txt")[:, 1:5], h, w)
		obj, bg = inside.nonzero().flatten().numpy(), (~inside).nonzero().flatten().numpy()
		sel = np.concatenate([
			rng.choice(obj, min(per_image_obj, len(obj)), replace=False) if len(obj) else np.zeros(0, int),
			rng.choice(bg, min(per_image_bg, len(bg)), replace=False) if len(bg) else np.zeros(0, int),
		]).astype(np.int64)
		pool.append(tok[sel])
		if verbose and (i + 1) % 100 == 0:
			print(f"[MEMORY] {i + 1}/{len(paths)} images")
	pool_t = torch.cat(pool).to(device)
	idx = greedy_coreset(pool_t, cfg.model.memory_size).to(device)
	bank = pool_t[idx].cpu()
	if verbose:
		print(f"[MEMORY] pool={pool_t.shape[0]} tokens -> bank {tuple(bank.shape)} from {len(paths)} images")
	return bank, memory_fingerprint(cfg, backbone, len(paths))


def validate_bank(bank: torch.Tensor, channels: int) -> None:
	if bank.ndim != 2 or bank.shape[1] != channels or bank.shape[0] == 0:
		raise ValueError(f"bank must be [K,{channels}], got {tuple(bank.shape)}")
	if not torch.isfinite(bank).all():
		raise ValueError("bank has non-finite values")
	if not torch.allclose(bank.norm(dim=1), torch.ones(bank.shape[0]), atol=1e-3):
		raise ValueError("bank tokens must be L2-normalised")


def get_memory_bank(backbone: ResNetBackbone, cfg: PatchDETRConfig, device: str, path: str,
					rebuild: bool = False) -> torch.Tensor:
	"""Loads the cached bank only when its fingerprint matches the current configuration."""
	n_images = len(subsample(list_images(cfg.data.memory_dir), cfg.data.limit))
	want = memory_fingerprint(cfg, backbone, n_images)
	if os.path.isfile(path) and not rebuild:
		payload = torch.load(path, map_location="cpu")
		if payload.get("fingerprint") == want:
			validate_bank(payload["memory_bank"], backbone.out_channels)
			print(f"[MEMORY] loaded {tuple(payload['memory_bank'].shape)} from {path}")
			return payload["memory_bank"]
		print("[MEMORY] cached bank does not match the current configuration; rebuilding")
	bank, fp = build_memory_bank(backbone, cfg, device)
	validate_bank(bank, backbone.out_channels)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save({"memory_bank": bank, "fingerprint": fp}, path)
	return bank
