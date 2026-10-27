"""Data-level diagnostics (no training needed):  python -m patchdetr.diagnostics [config flags]

Checks, on the real dataset, that
  * the memory bank separates normal tokens from degraded ones (E larger on mud-covered objects),
  * the temporal prior is active and where its energy sits relative to labelled objects.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from .config import PatchDETRConfig, config_from_args
from .data import make_datasets
from .memory import get_memory_bank, object_token_mask
from .model import PatchDETR


@torch.no_grad()
def run(cfg: PatchDETRConfig, n: int = 40) -> Dict[str, float]:
	device = cfg.resolve_device()
	os.makedirs(cfg.train.output_dir, exist_ok=True)
	model = PatchDETR(cfg).to(device).eval()
	bank = get_memory_bank(model.backbone, cfg, device, os.path.join(cfg.train.output_dir, "memory_bank.pt"))
	model.patchcvt.set_memory(bank.to(device))
	ds = make_datasets(cfg)
	clean, deg = ds["val_clean"], ds["val_degraded"]
	idx = np.linspace(0, len(clean) - 1, min(n, len(clean))).astype(int)
	acc: Dict[str, list] = {k: [] for k in ("E_obj_clean", "E_obj_deg", "E_bg_clean", "E_bg_deg", "motion_obj", "motion_bg", "motion_active", "motion_mean")}
	for i in idx:
		c, d = clean[i], deg[i]
		if len(c["boxes"]) == 0:
			continue
		ec, ed = (model.encode(s["clean"][None].to(device)) for s in (c, d))
		h, w = ec["hw"]
		m = object_token_mask(c["boxes"].numpy(), h, w).to(device)
		for name, enc in (("clean", ec), ("deg", ed)):
			acc[f"E_obj_{name}"].append(float(enc["E"][0][m].mean()))
			acc[f"E_bg_{name}"].append(float(enc["E"][0][~m].mean()))
		mh, mw = c["motion"].shape[-2:]
		mm = object_token_mask(c["boxes"].numpy(), mh, mw).reshape(mh, mw)
		mo = c["motion"][0]
		acc["motion_active"].append(float(c["has_motion"]))
		if c["has_motion"]:
			acc["motion_obj"].append(float(mo[mm].mean()))
			acc["motion_bg"].append(float(mo[~mm].mean()))
			acc["motion_mean"].append(float(mo.mean()))
	res = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
	for k, v in res.items():
		print(f"{k:16s} {v:.4f}")
	return res


if __name__ == "__main__":
	run(config_from_args())
