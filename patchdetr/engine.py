"""Trainer: usable from the CLI (train.py) and from a notebook."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import PatchDETRConfig
from .data import make_datasets, make_loader
from .evaluation import evaluate, prior_diagnostics
from .losses import PatchDETRCriterion
from .memory import build_memory_bank, get_memory_bank
from .model import PatchDETR


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


class Trainer:
	"""Typical notebook use:
	    t = Trainer(cfg); t.setup(); t.fit(); t.test()
	Model selection uses ONLY the validation split carved from the train folder; the test split
	is touched exclusively by `test()` (called once after training, on the best checkpoint)."""

	def __init__(self, cfg: PatchDETRConfig) -> None:
		cfg.validate()
		self.cfg = cfg
		self.device = cfg.resolve_device()
		self.amp = cfg.train.amp and self.device == "cuda"
		self.history: List[Dict[str, Any]] = []
		self.best_metric = -1.0
		self.start_epoch = 0
		self.model: Optional[PatchDETR] = None
		self.datasets: Dict[str, Any] = {}
		os.makedirs(cfg.train.output_dir, exist_ok=True)

	# ---- setup ----------------------------------------------------------------------------
	def setup(self) -> "Trainer":
		cfg = self.cfg
		seed_everything(cfg.train.seed)
		self.datasets = make_datasets(cfg)
		self.model = PatchDETR(cfg).to(self.device)
		bank = get_memory_bank(self.model.backbone, cfg, self.device, self.bank_path, cfg.train.rebuild_memory)
		self.model.patchcvt.set_memory(bank.to(self.device))
		self.criterion = PatchDETRCriterion(cfg.loss)
		backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
		bb_ids = {id(p) for p in backbone_params}
		rest = [p for p in self.model.parameters() if p.requires_grad and id(p) not in bb_ids]
		groups = [{"params": rest, "lr": cfg.train.lr}]
		if backbone_params:
			groups.append({"params": backbone_params, "lr": cfg.train.lr * cfg.train.backbone_lr_mult})
		self.params = rest + backbone_params
		self.optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.train.epochs)
		self.scaler = torch.amp.GradScaler(self.device, enabled=self.amp)
		self.loaders = {
			"train": make_loader(self.datasets["train"], cfg, True, self.device),
			**{k: make_loader(v, cfg, False, self.device) for k, v in self.datasets.items() if k != "train"},
		}
		cfg.save(os.path.join(cfg.train.output_dir, "config.json"))
		if cfg.train.resume:
			self.load(cfg.train.resume, resume=True)
		n = sum(p.numel() for p in self.params)
		print(f"[SETUP] device={self.device} amp={self.amp} trainable={n / 1e6:.2f}M "
			  + " ".join(f"{k}={len(v)}" for k, v in self.datasets.items()))
		return self

	@property
	def bank_path(self) -> str:
		return os.path.join(self.cfg.train.output_dir, "memory_bank.pt")

	# ---- training -------------------------------------------------------------------------
	def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
		m, d = self.model, self.device
		clean, degraded, motion = batch["clean"].to(d), batch["degraded"].to(d), batch["motion"].to(d)
		with torch.autocast(d, dtype=torch.float16, enabled=self.amp):
			out_c, out_d = m(clean, motion), m(degraded, motion)
		loss, logs = self.criterion(out_c, out_d, batch["targets"])
		self.optimizer.zero_grad(set_to_none=True)
		self.scaler.scale(loss).backward()
		self.scaler.unscale_(self.optimizer)
		nn.utils.clip_grad_norm_(self.params, self.cfg.train.clip)
		self.scaler.step(self.optimizer)
		self.scaler.update()
		logs["motion_active"] = float(batch["has_motion"].float().mean())
		return logs

	def train_epoch(self) -> Dict[str, float]:
		self.model.train()
		agg: Dict[str, float] = {}
		steps = 0
		for batch in self.loaders["train"]:
			for k, v in self.train_step(batch).items():
				agg[k] = agg.get(k, 0.0) + v
			steps += 1
		self.scheduler.step()
		return {k: v / max(steps, 1) for k, v in agg.items()}

	def evaluate(self, split: str) -> Dict[str, float]:
		"""split in {val, test}: returns clean + degraded metrics (+ their mean AP50)."""
		res = {}
		for kind in ("clean", "degraded"):
			res.update(evaluate(self.model, self.loaders[f"{split}_{kind}"], self.cfg, self.device, f"{split}_{kind}_"))
		res[f"{split}_ap50_mean"] = 0.5 * (res[f"{split}_clean_ap50"] + res[f"{split}_degraded_ap50"])
		return res

	def diagnostics(self, split: str = "val_clean") -> Dict[str, float]:
		return prior_diagnostics(self.model, self.loaders[split], self.device)

	def refresh_memory(self) -> None:
		"""Rebuild M with the current backbone (only meaningful when train_backbone=True)."""
		bank, fp = build_memory_bank(self.model.backbone, self.cfg, self.device, verbose=False)
		self.model.patchcvt.set_memory(bank.to(self.device))
		torch.save({"memory_bank": bank, "fingerprint": fp}, self.bank_path)

	def fit(self, epochs: Optional[int] = None, callback=None) -> List[Dict[str, Any]]:
		cfg = self.cfg
		total = epochs or cfg.train.epochs
		log_path = os.path.join(cfg.train.output_dir, "train_log.csv")
		for epoch in range(self.start_epoch, total):
			t0 = time.time()
			tr = self.train_epoch()
			if cfg.model.train_backbone and cfg.model.memory_refresh_every and (epoch + 1) % cfg.model.memory_refresh_every == 0:
				self.refresh_memory()
			ev = self.evaluate("val")
			row = {"epoch": epoch + 1, "time": time.time() - t0, **{f"train_{k}": v for k, v in tr.items()}, **ev}
			self.history.append(row)
			print(f"[EPOCH {epoch + 1}/{total}] {row['time']:.0f}s loss={tr['loss']:.3f} det={tr['cls']:.3f}/{tr['l1']:.3f}/{tr['giou']:.3f} "
				  f"val_ap50 clean={ev['val_clean_ap50']:.3f} degraded={ev['val_degraded_ap50']:.3f}")
			new = not os.path.isfile(log_path)
			with open(log_path, "a", newline="") as f:
				w = csv.DictWriter(f, fieldnames=list(row))
				if new:
					w.writeheader()
				w.writerow(row)
			metric = ev[cfg.train.selection_metric]
			self.save(os.path.join(cfg.train.output_dir, "last.pt"), epoch)
			if metric > self.best_metric:
				self.best_metric = metric
				self.save(os.path.join(cfg.train.output_dir, "best.pt"), epoch)
			if callback:
				callback(self, row)
		return self.history

	def test(self, checkpoint: Optional[str] = None) -> Dict[str, float]:
		"""One-shot evaluation of the best (validation-selected) checkpoint on the test split."""
		self.load(checkpoint or os.path.join(self.cfg.train.output_dir, "best.pt"))
		res = self.evaluate("test")
		res["prior_diagnostics"] = prior_diagnostics(self.model, self.loaders["test_clean"], self.device)
		with open(os.path.join(self.cfg.train.output_dir, "test_results.json"), "w") as f:
			json.dump(res, f, indent=2)
		return res

	# ---- checkpoints ----------------------------------------------------------------------
	def save(self, path: str, epoch: int) -> None:
		torch.save({"model": self.model.state_dict(), "opt": self.optimizer.state_dict(),
					"sched": self.scheduler.state_dict(), "epoch": epoch, "best_metric": self.best_metric,
					"history": self.history, "config": self.cfg.to_dict()}, path)

	def load(self, path: str, resume: bool = False) -> None:
		ck = torch.load(path, map_location=self.device, weights_only=False)
		self.model.load_state_dict(ck["model"])
		if resume:
			self.optimizer.load_state_dict(ck["opt"])
			self.scheduler.load_state_dict(ck["sched"])
			self.start_epoch, self.best_metric, self.history = ck["epoch"] + 1, ck["best_metric"], ck["history"]
