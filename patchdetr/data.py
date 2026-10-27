"""Datasets, YOLO IO, sequence-aware train/val split, temporal (motion) prior."""
from __future__ import annotations

import os
import re
import zlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from .config import PatchDETRConfig
from .degradation import MudDegradation

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_FRAME_RE = re.compile(r"^(.+_)(\d{6})_(\d{2}-\d{2}-\d{2})\.\w+$")
MOTION_DOWNSCALE = 4  # motion prior is stored at (H/4, W/4) and pooled to the token grid


def read_yolo(label_path: str) -> np.ndarray:
	"""[n,5] array of class, cx, cy, w, h (normalised); tolerant to CRLF files."""
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
	return sorted(os.path.join(d, n) for d, _, ns in os.walk(folder) for n in ns
				  if os.path.splitext(n)[1].lower() in VALID_EXT)


def subsample(paths: Sequence[str], limit: Optional[int]) -> List[str]:
	"""Evenly spaced subset (a head slice would only contain one sequence)."""
	if not limit or limit >= len(paths):
		return list(paths)
	return [paths[int(i)] for i in np.linspace(0, len(paths) - 1, limit)]


def frame_key(path: str) -> Tuple[str, int]:
	"""(sequence id, frame number). Un-patterned files (e.g. 12.jpg) form one 'misc' sequence."""
	m = _FRAME_RE.match(os.path.basename(path))
	if m:
		return os.path.dirname(path) + "/" + m.group(1), int(m.group(2))
	stem = os.path.splitext(os.path.basename(path))[0]
	return os.path.dirname(path) + "/misc", int(stem) if stem.isdigit() else 0


def build_frame_index(paths: Sequence[str]) -> Dict[Tuple[str, int], str]:
	return {frame_key(p): p for p in paths}


def split_train_val(paths: Sequence[str], val_fraction: float, gap: int) -> Tuple[List[str], List[str]]:
	"""Contiguous-block split per sequence: the last `val_fraction` of every sequence is
	validation and `gap` frames before it are discarded, so near-identical adjacent
	frames never straddle the split."""
	groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
	for p in paths:
		seq, idx = frame_key(p)
		groups[seq].append((idx, p))
	train, val = [], []
	for items in groups.values():
		items.sort()
		n_val = int(round(len(items) * val_fraction))
		n_train = max(len(items) - n_val - (gap if n_val else 0), 0)
		train += [p for _, p in items[:n_train]]
		val += [p for _, p in items[len(items) - n_val:]]
	return sorted(train), sorted(val)


def load_image(path: str, size: Tuple[int, int]) -> Image.Image:
	"""Load + resize to (W,H); JPEG draft mode makes 4K frames cheap to decode."""
	with Image.open(path) as im:
		im.draft("RGB", (size[0] * 2, size[1] * 2))
		return im.convert("RGB").resize(size, Image.BILINEAR)


def to_tensor(im: Image.Image) -> torch.Tensor:
	return TF.normalize(TF.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD)


def cxcywh_to_xyxy_np(b: np.ndarray, w: int, h: int) -> np.ndarray:
	if len(b) == 0:
		return np.zeros((0, 4), np.float32)
	return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
					 (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)


class DetectionDataset(Dataset):
	"""mode: 'train'  -> random flip + clean/degraded pair (random degradation)
	        'eval'   -> clean image only
	        'eval_degraded' -> deterministically degraded image (seeded by file path)"""

	def __init__(self, paths: Sequence[str], cfg: PatchDETRConfig, mode: str,
				 frame_index: Optional[Dict[Tuple[str, int], str]] = None) -> None:
		assert mode in {"train", "eval", "eval_degraded"}
		self.paths, self.mode = list(paths), mode
		self.size = (cfg.data.img_w, cfg.data.img_h)
		self.use_motion = cfg.data.use_motion
		self.frame_index = frame_index if frame_index is not None else build_frame_index(self.paths)
		self.degrade = MudDegradation(cfg.data.mud_prob)

	def __len__(self) -> int:
		return len(self.paths)

	def previous_frame(self, path: str) -> Optional[str]:
		seq, idx = frame_key(path)
		if seq.endswith("/misc"):
			return None
		return self.frame_index.get((seq, idx - 1))

	def motion_prior(self, path: str, image: Image.Image) -> Tuple[np.ndarray, bool]:
		"""P_motion (eq. 32): normalised |I_t - I_{t-1}| against the previous frame of the same
		sequence. Always returns the same shape; all zeros (and False) if no previous frame."""
		W, H = self.size
		shape = (H // MOTION_DOWNSCALE, W // MOTION_DOWNSCALE)
		prev = self.previous_frame(path) if self.use_motion else None
		if prev is None:
			return np.zeros(shape, np.float32), False
		small = (shape[1], shape[0])
		a = np.asarray(image.convert("L").resize(small), dtype=np.float32)
		b = np.asarray(load_image(prev, self.size).convert("L").resize(small), dtype=np.float32)
		d = np.abs(a - b)
		d = np.asarray(Image.fromarray(d.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
		d = d / max(float(np.percentile(d, 99.5)), 8.0)
		return np.clip(d, 0, 1).astype(np.float32), True

	def __getitem__(self, i: int) -> Dict[str, Any]:
		path = self.paths[i]
		image = load_image(path, self.size)
		lab = read_yolo(os.path.splitext(path)[0] + ".txt")
		motion, has_motion = self.motion_prior(path, image)
		boxes = lab[:, 1:5].copy()
		if self.mode == "train":
			rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
		else:
			rng = np.random.default_rng(zlib.crc32(path.encode()))
		if self.mode == "train" and rng.random() < 0.5:
			image = image.transpose(Image.FLIP_LEFT_RIGHT)
			motion = motion[:, ::-1].copy()
			boxes[:, 0] = 1 - boxes[:, 0]
		xyxy = cxcywh_to_xyxy_np(boxes, *self.size)
		sample: Dict[str, Any] = {"motion": torch.from_numpy(motion)[None], "has_motion": has_motion,
								  "boxes": torch.from_numpy(boxes),
								  "labels": torch.zeros(len(boxes), dtype=torch.long), "path": path}
		if self.mode == "train":
			sample["clean"], sample["degraded"] = to_tensor(image), to_tensor(self.degrade(image, xyxy, rng))
		elif self.mode == "eval_degraded":
			sample["clean"], sample["degraded"] = to_tensor(self.degrade(image, xyxy, rng)), None
		else:
			sample["clean"], sample["degraded"] = to_tensor(image), None
		return sample


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
	return {
		"clean": torch.stack([b["clean"] for b in batch]),
		"degraded": torch.stack([b["degraded"] for b in batch]) if batch[0]["degraded"] is not None else None,
		"motion": torch.stack([b["motion"] for b in batch]),
		"has_motion": torch.tensor([b["has_motion"] for b in batch]),
		"targets": [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch],
		"paths": [b["path"] for b in batch],
	}


def make_datasets(cfg: PatchDETRConfig) -> Dict[str, Dataset]:
	"""train / val_clean / val_degraded (from the train folder) and test_clean / test_degraded."""
	train_all = list_images(cfg.data.train_dir)
	train_paths, val_paths = split_train_val(train_all, cfg.data.val_fraction, cfg.data.val_gap)
	train_paths, val_paths = subsample(train_paths, cfg.data.limit), subsample(val_paths, cfg.data.limit)
	test_all = list_images(cfg.data.test_dir)
	test_paths = subsample(test_all, cfg.data.limit)
	fi_train, fi_test = build_frame_index(train_all), build_frame_index(test_all)
	return {
		"train": DetectionDataset(train_paths, cfg, "train", fi_train),
		"val_clean": DetectionDataset(val_paths, cfg, "eval", fi_train),
		"val_degraded": DetectionDataset(val_paths, cfg, "eval_degraded", fi_train),
		"test_clean": DetectionDataset(test_paths, cfg, "eval", fi_test),
		"test_degraded": DetectionDataset(test_paths, cfg, "eval_degraded", fi_test),
	}


def make_loader(ds: Dataset, cfg: PatchDETRConfig, shuffle: bool, device: str = "cpu") -> DataLoader:
	w = cfg.data.workers
	return DataLoader(ds, cfg.train.batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) >= cfg.train.batch_size,
					  num_workers=w, collate_fn=collate, pin_memory=device == "cuda", persistent_workers=w > 0)
