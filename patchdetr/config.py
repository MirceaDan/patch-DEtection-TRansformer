"""Configuration dataclasses shared by the CLI, the notebook and the tests."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, get_type_hints

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataConfig:
	train_dir: str = os.path.join(ROOT, "database", "train")
	test_dir: str = os.path.join(ROOT, "database", "test")
	memory_dir: str = os.path.join(ROOT, "memorybank")
	img_w: int = 1024
	img_h: int = 576
	val_fraction: float = 0.15  # held out from train (contiguous block per sequence)
	val_gap: int = 5  # frames dropped between train and val blocks (near-duplicate frames)
	workers: int = 4
	mud_prob: float = 0.7
	use_motion: bool = True
	limit: Optional[int] = None  # cap images per split (debug only; takes evenly spaced samples)


@dataclass
class ModelConfig:
	pretrained: bool = True
	stride8: bool = False  # ResNet18 up to layer2 (stride 8) instead of layer3 (stride 16)
	dim: int = 256
	heads: int = 8
	enc_layers: int = 2
	dec_layers: int = 4
	num_queries: int = 50
	dropout: float = 0.1
	tau: float = 0.1  # retrieval temperature (eq. 22)
	memory_size: int = 2048
	train_backbone: bool = False  # unfreeze layer3 so that L_memory has a gradient path
	memory_refresh_every: int = 0  # epochs between memory rebuilds when train_backbone (0 = never)


@dataclass
class LossConfig:
	w_cls: float = 1.0
	w_l1: float = 5.0
	w_giou: float = 2.0
	w_heat: float = 1.0
	eos_coef: float = 0.1
	w_pcvt: float = 1.0  # weight of the whole PatchCvT objective (eq. 22)
	lam_semantic: float = 1.0
	lam_memory: float = 0.5
	lam_robust: float = 0.25
	lam_gate: float = 0.25
	# Documented deviation (see README_PATCHDETR.md, "Gate target conflict"):
	# extra term mean_{j in objects}(1 - S_j). 0.0 reproduces the exact formulation.
	gate_preserve: float = 0.25
	semantic_detach_clean: bool = False


@dataclass
class TrainConfig:
	epochs: int = 50
	batch_size: int = 2
	lr: float = 1e-4
	backbone_lr_mult: float = 0.1
	weight_decay: float = 1e-4
	clip: float = 0.1
	seed: int = 23
	device: str = "auto"
	amp: bool = True
	output_dir: str = os.path.join(ROOT, "model")
	selection_metric: str = "val_ap50_mean"  # mean of clean and degraded validation AP50
	rebuild_memory: bool = False
	resume: Optional[str] = None
	run_test: bool = True  # evaluate best checkpoint on the test split ONCE after training


@dataclass
class PatchDETRConfig:
	data: DataConfig = field(default_factory=DataConfig)
	model: ModelConfig = field(default_factory=ModelConfig)
	loss: LossConfig = field(default_factory=LossConfig)
	train: TrainConfig = field(default_factory=TrainConfig)

	def to_dict(self) -> Dict[str, Any]:
		return dataclasses.asdict(self)

	@classmethod
	def from_dict(cls, d: Dict[str, Any]) -> "PatchDETRConfig":
		kw = {}
		for f in dataclasses.fields(cls):
			sub = f.default_factory  # type: ignore[misc]
			valid = {x.name for x in dataclasses.fields(sub)}
			kw[f.name] = sub(**{k: v for k, v in d.get(f.name, {}).items() if k in valid})
		return cls(**kw)

	def save(self, path: str) -> None:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(self.to_dict(), f, indent=2)

	def resolve_device(self) -> str:
		if self.train.device != "auto":
			return self.train.device
		import torch
		return "cuda" if torch.cuda.is_available() else "cpu"

	@property
	def stride(self) -> int:
		return 8 if self.model.stride8 else 16

	def validate(self) -> None:
		if self.data.img_w % self.stride or self.data.img_h % self.stride:
			raise ValueError(f"img_w/img_h must be multiples of {self.stride}")
		if self.model.dim % self.model.heads or self.model.dim % 4:
			raise ValueError("dim must be divisible by heads and by 4")
		if self.model.train_backbone and self.model.stride8:
			raise ValueError("train_backbone requires the layer3 backbone (stride 16)")
		if not 0 <= self.data.val_fraction < 1:
			raise ValueError("val_fraction must be in [0,1)")


def add_dataclass_args(parser: argparse.ArgumentParser, cfg: PatchDETRConfig) -> None:
	"""Exposes every config field as a --flag (field names are unique across sections)."""
	for section in dataclasses.fields(cfg):
		obj = getattr(cfg, section.name)
		hints = get_type_hints(type(obj))
		for f in dataclasses.fields(obj):
			default, name = getattr(obj, f.name), "--" + f.name.replace("_", "-")
			typ = hints[f.name]
			if typ is bool:
				parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
			else:
				base = {int: int, float: float, str: str, Optional[int]: int, Optional[str]: str}[typ]
				parser.add_argument(name, type=base, default=default)


def config_from_args(argv: Optional[Sequence[str]] = None) -> PatchDETRConfig:
	cfg = PatchDETRConfig()
	parser = argparse.ArgumentParser(description="Train PatchDETR (PatchCvT-modulated DETR)")
	parser.add_argument("--exact-formulation", action="store_true",
						help="disable documented deviations (gate_preserve=0)")
	add_dataclass_args(parser, cfg)
	ns = vars(parser.parse_args(argv))
	exact = ns.pop("exact_formulation")
	for section in dataclasses.fields(cfg):
		obj = getattr(cfg, section.name)
		for f in dataclasses.fields(obj):
			setattr(obj, f.name, ns[f.name])
	if exact:
		cfg.loss.gate_preserve = 0.0
	cfg.validate()
	return cfg
