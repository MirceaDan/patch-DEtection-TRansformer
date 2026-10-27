"""Synthetic degradation (mud / dirt / blur / lighting / JPEG) that preserves geometry,
so labels remain valid and clean/degraded tokens stay spatially aligned."""
from __future__ import annotations

import io
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


def mud_patch(box: Tuple[int, int, int, int], rng: np.random.Generator,
			  coverage: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
	"""Returns (alpha [h,w,1], colour [h,w,3]) of an irregular mud blob inside `box`."""
	x0, y0, x1, y1 = box
	h, w = max(1, y1 - y0), max(1, x1 - x0)
	coverage = float(rng.uniform(0.3, 1.0)) if coverage is None else coverage
	yy, xx = np.mgrid[0:h, 0:w]
	cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
	dist = ((xx - cx) / (w * coverage)) ** 2 + ((yy - cy) / (h * coverage)) ** 2
	small = rng.random((h // 3 + 1, w // 3 + 1))
	noise = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255.0
	alpha = np.clip((1.2 - dist) * 2.0 + (noise - 0.5), 0, 1) * rng.uniform(0.7, 1.0)
	base = np.array([rng.integers(70, 120), rng.integers(50, 90), rng.integers(30, 60)], dtype=np.float32)
	colour = base[None, None, :] * (0.7 + 0.6 * noise[..., None])
	return alpha[..., None].astype(np.float32), colour.astype(np.float32)


class MudDegradation:
	def __init__(self, mud_prob: float = 0.7) -> None:
		self.mud_prob = mud_prob

	def __call__(self, image: Image.Image, boxes_xyxy: np.ndarray, rng: np.random.Generator) -> Image.Image:
		if rng.random() < 0.5:
			image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.5))))
		if rng.random() < 0.7:
			image = TF.adjust_brightness(image, float(rng.uniform(0.5, 1.4)))
			image = TF.adjust_contrast(image, float(rng.uniform(0.5, 1.4)))
			image = TF.adjust_saturation(image, float(rng.uniform(0.6, 1.2)))
		arr = np.asarray(image).astype(np.float32)
		H, W = arr.shape[:2]
		for b in boxes_xyxy:
			if rng.random() > self.mud_prob:
				continue
			x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
			x1, y1 = min(W, int(math.ceil(b[2]))), min(H, int(math.ceil(b[3])))
			if x1 - x0 < 2 or y1 - y0 < 2:
				continue
			alpha, colour = mud_patch((x0, y0, x1, y1), rng)
			arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1 - alpha) + colour * alpha
		image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
		if rng.random() < 0.6:
			buf = io.BytesIO()
			image.save(buf, "JPEG", quality=int(rng.integers(15, 60)))
			buf.seek(0)
			image = Image.open(buf).convert("RGB")
		return image
