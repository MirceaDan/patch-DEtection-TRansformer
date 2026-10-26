from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18


################################################################################
# CONFIGURATION / CONSTANTS
################################################################################

PROJECT_ROOT = r"E:\Projects\patch-convolutional-vision-transformer"
TRAIN_DATASET = os.path.join(PROJECT_ROOT, "database", "train")
TEST_DATASET = os.path.join(PROJECT_ROOT, "database", "test")
MEMORY_BANK_DATASET = os.path.join(PROJECT_ROOT, "memorybank")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "model")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last.pt")
MEMORY_BANK_PATH = os.path.join(CHECKPOINT_DIR, "memory_bank.pt")
EVALUATION_CSV_PATH = os.path.join(CHECKPOINT_DIR, "test_scores.csv")

IMAGE_SIZE = 224
INPUT_CHANNELS = 3
CNN_CHANNELS = 128
FEATURE_MAP_SIZE = 28
NUM_SPATIAL_TOKENS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE

BATCH_SIZE = 1
MEMORY_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 0
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_VALIDATION_FRACTION = 0.1

# The archived experiment used ImageNet-pretrained, frozen ResNet18 features.
# Set False to use a randomly initialized frozen backbone without downloading.
USE_PRETRAINED_RESNET18 = True
TRANSFORMER_DIM = 256
TRANSFORMER_DEPTH = 2
NUM_ATTENTION_HEADS = 4
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_FEEDFORWARD_DIM = 1024

MEMORY_BANK_SIZE = 512
TOP_K_PATCHES = 16
AGGREGATION_MODE = "max"

# Bounded post-processing for the requested viability display only:
# viability = exp(-raw_squared_distance / scale). Raw NN distances are always
# also retained/reported. This is a deterministic distance transform, not a
# probability, learned head, or part of the Transformer architecture.
DISTANCE_TO_VIABILITY_SCALE = 1.0
VALIDITY_THRESHOLD: Optional[float] = 0.5  # Evaluation-only heuristic.

RANDOM_SEED = 23
DEBUG_SHAPES = True
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################
# REPRODUCIBILITY
################################################################################

def set_random_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
	worker_seed = torch.initial_seed() % (2**32)
	np.random.seed(worker_seed)
	random.seed(worker_seed)


################################################################################
# DATASET / YOLO ROI LOADING
################################################################################

def parse_yolo_annotation(label_path: str) -> List[Dict[str, float]]:
	annotations: List[Dict[str, float]] = []
	if not os.path.exists(label_path):
		return annotations

	with open(label_path, "r", encoding="utf-8") as label_file:
		lines = label_file.read().splitlines()
	for line_number, line in enumerate(lines, 1):
		values = line.strip().split()
		if not values:
			continue
		if len(values) < 5:
			raise ValueError(f"Expected YOLO class/cx/cy/w/h at {label_path}:{line_number}")
		try:
			class_id = int(float(values[0]))
			center_x, center_y, box_width, box_height = map(float, values[1:5])
		except ValueError as exc:
			raise ValueError(f"Invalid YOLO values at {label_path}:{line_number}") from exc
		if not all(math.isfinite(value) for value in (center_x, center_y, box_width, box_height)):
			raise ValueError(f"Non-finite YOLO coordinates at {label_path}:{line_number}")
		if box_width <= 0 or box_height <= 0:
			raise ValueError(f"YOLO box has non-positive size at {label_path}:{line_number}")
		annotations.append(
			{
				"class_id": class_id,
				"center_x": center_x,
				"center_y": center_y,
				"width": box_width,
				"height": box_height,
			}
		)
	return annotations


def collect_roi_samples(dataset_path: str) -> List[Dict[str, Any]]:
	if not os.path.isdir(dataset_path):
		raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

	image_paths = sorted(
		os.path.join(directory, filename)
		for directory, _, filenames in os.walk(dataset_path)
		for filename in filenames
		if os.path.isfile(os.path.join(directory, filename))
		and os.path.splitext(filename)[1].lower() in VALID_IMAGE_EXTENSIONS
	)
	samples: List[Dict[str, Any]] = []
	missing_annotations = 0
	for image_path in image_paths:
		label_path = os.path.splitext(image_path)[0] + ".txt"
		if not os.path.isfile(label_path):
			missing_annotations += 1
			continue
		for object_index, annotation in enumerate(parse_yolo_annotation(label_path)):
			samples.append(
				{
					"image_path": image_path,
					"label_path": label_path,
					"object_index": object_index,
					**annotation,
				}
			)

	if missing_annotations:
		print(f"[DATA] {dataset_path}: skipped {missing_annotations} images without YOLO labels")
	print(f"[DATA] {dataset_path}: {len(image_paths)} images, {len(samples)} annotated ROIs")
	return samples


def crop_yolo_roi(image: Image.Image, sample: Dict[str, Any]) -> Image.Image:
	image_width, image_height = image.size
	center_x = float(sample["center_x"]) * image_width
	center_y = float(sample["center_y"]) * image_height
	box_width = float(sample["width"]) * image_width
	box_height = float(sample["height"]) * image_height

	left = max(0, min(image_width, int(math.floor(center_x - box_width / 2))))
	top = max(0, min(image_height, int(math.floor(center_y - box_height / 2))))
	right = max(0, min(image_width, int(math.ceil(center_x + box_width / 2))))
	bottom = max(0, min(image_height, int(math.ceil(center_y + box_height / 2))))
	if right <= left or bottom <= top:
		raise ValueError(f"YOLO box crops to an empty ROI: {sample['label_path']} object {sample['object_index']}")
	return image.crop((left, top, right, bottom))


class ROIPairDataset(Dataset):
	def __init__(
		self,
		samples: Sequence[Dict[str, Any]],
		eval_transform: T.Compose,
		degradation_transform: Optional[Any] = None,
	) -> None:
		self.samples = list(samples)
		self.eval_transform = eval_transform
		self.degradation_transform = degradation_transform

	def __len__(self) -> int:
		return len(self.samples)

	def get_roi(self, index: int) -> Image.Image:
		sample = self.samples[index]
		with Image.open(sample["image_path"]) as source:
			image = source.convert("RGB")
		return crop_yolo_roi(image, sample)

	def __getitem__(self, index: int) -> Dict[str, Any]:
		sample = self.samples[index]
		roi = self.get_roi(index)
		clean = self.eval_transform(roi)
		if self.degradation_transform is not None:
			degraded_roi = self.degradation_transform(roi.copy())
			degraded = self.eval_transform(degraded_roi)
			return {"clean": clean, "degraded": degraded}
		return {
			"image": clean,
			"image_path": str(sample["image_path"]),
			"object_index": sample["object_index"],
			"class_id": sample["class_id"],
		}


def build_loader(
	dataset: Dataset,
	batch_size: int,
	shuffle: bool,
) -> DataLoader:
	generator = torch.Generator().manual_seed(RANDOM_SEED)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=NUM_WORKERS,
		pin_memory=DEVICE.type == "cuda",
		worker_init_fn=seed_worker,
		generator=generator,
	)


################################################################################
# AUGMENTATIONS / PREPROCESSING
################################################################################

class RandomJPEGCompression:
	def __call__(self, image: Image.Image) -> Image.Image:
		quality = random.randint(10, 60)
		image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
		success, encoded = cv2.imencode(
			".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
		)
		if not success:
			raise RuntimeError("OpenCV failed to encode a JPEG degradation")
		decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
		return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


class RandomOcclusion:
	def __init__(self, probability: float = 0.5) -> None:
		self.probability = probability

	def __call__(self, image: Image.Image) -> Image.Image:
		if random.random() >= self.probability:
			return image
		array = np.asarray(image).copy()
		height, width = array.shape[:2]
		occlusion_width = random.randint(max(1, int(0.1 * width)), max(1, int(0.4 * width)))
		occlusion_height = random.randint(max(1, int(0.1 * height)), max(1, int(0.4 * height)))
		left = random.randint(0, max(0, width - occlusion_width))
		top = random.randint(0, max(0, height - occlusion_height))
		array[top:top + occlusion_height, left:left + occlusion_width] = 0
		return Image.fromarray(array)


def build_degradation_transform() -> T.Compose:
	return T.Compose(
		[
			T.RandomApply([T.GaussianBlur(kernel_size=7)], p=0.5),
			T.RandomApply(
				[T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2)],
				p=0.7,
			),
			RandomJPEGCompression(),
			RandomOcclusion(probability=0.5),
			T.RandomPerspective(distortion_scale=0.3, p=0.3),
		]
	)


def build_eval_transform() -> T.Compose:
	return T.Compose(
		[
			T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			T.ToTensor(),
			T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)


def build_validation_degradation() -> T.Compose:
	return T.GaussianBlur(kernel_size=7)


################################################################################
# RESNET18 CNN TOKEN EXTRACTOR
################################################################################

class ResNet18SpatialExtractor(nn.Module):
	"""Frozen ResNet18 stem + layer1 + layer2, whose stride yields 28x28."""

	def __init__(self, pretrained: bool = USE_PRETRAINED_RESNET18) -> None:
		super().__init__()
		weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
		backbone = resnet18(weights=weights)
		self.stem = nn.Sequential(
			backbone.conv1,
			backbone.bn1,
			backbone.relu,
			backbone.maxpool,
		)
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		self.output_channels = 128
		for parameter in self.parameters():
			parameter.requires_grad = False
		self.eval()

	def train(self, mode: bool = True) -> "ResNet18SpatialExtractor":
		super().train(False)
		return self

	@torch.no_grad()
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		features = self.stem(images)
		features = self.layer1(features)
		features = self.layer2(features)
		if tuple(features.shape[-2:]) != (FEATURE_MAP_SIZE, FEATURE_MAP_SIZE):
			raise RuntimeError(
				"ResNet18 spatial output must be 28x28 for 224x224 ROIs; "
				f"got {tuple(features.shape[-2:])}"
			)
		if features.shape[1] != self.output_channels:
			raise RuntimeError(f"Expected 128 ResNet18 channels, got {features.shape[1]}")
		return features


def feature_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
	if feature_map.ndim != 4:
		raise ValueError(f"Expected feature map [B,C,H,W], got {tuple(feature_map.shape)}")
	if tuple(feature_map.shape[-2:]) != (FEATURE_MAP_SIZE, FEATURE_MAP_SIZE):
		raise RuntimeError(f"Expected 28x28 feature map, got {tuple(feature_map.shape[-2:])}")
	tokens = feature_map.flatten(start_dim=2).transpose(1, 2).contiguous()
	if tokens.shape[1] != NUM_SPATIAL_TOKENS:
		raise RuntimeError(f"Expected 784 CNN tokens, got {tokens.shape[1]}")
	return tokens


################################################################################
# TRANSFORMER
################################################################################

class SpatialTransformerEncoder(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.projection = nn.Linear(CNN_CHANNELS, TRANSFORMER_DIM)
		self.position_embedding = nn.Parameter(
			torch.zeros(1, NUM_SPATIAL_TOKENS, TRANSFORMER_DIM)
		)
		nn.init.trunc_normal_(self.position_embedding, std=0.02)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=TRANSFORMER_DIM,
			nhead=NUM_ATTENTION_HEADS,
			dim_feedforward=TRANSFORMER_FEEDFORWARD_DIM,
			dropout=TRANSFORMER_DROPOUT,
			activation="gelu",
			batch_first=True,
			norm_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_DEPTH)

	def forward(self, cnn_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if cnn_tokens.ndim != 3 or cnn_tokens.shape[1:] != (NUM_SPATIAL_TOKENS, CNN_CHANNELS):
			raise RuntimeError(
				f"Expected CNN tokens [B,784,{CNN_CHANNELS}], got {tuple(cnn_tokens.shape)}"
			)
		projected = self.projection(cnn_tokens)
		transformer_input = projected + self.position_embedding
		contextual_tokens = self.encoder(transformer_input)
		if contextual_tokens.shape[1] != NUM_SPATIAL_TOKENS:
			raise RuntimeError("Transformer changed the 784-token sequence length")
		if contextual_tokens.shape[2] != TRANSFORMER_DIM:
			raise RuntimeError("Transformer output dimension does not match D")
		return transformer_input, contextual_tokens


################################################################################
# PATCHCVT MODEL
################################################################################

class PatchCvT(nn.Module):
	def __init__(self, pretrained_resnet18: bool = USE_PRETRAINED_RESNET18) -> None:
		super().__init__()
		self.backbone = ResNet18SpatialExtractor(pretrained=pretrained_resnet18)
		self.transformer = SpatialTransformerEncoder()

	def forward(self, roi: torch.Tensor) -> Dict[str, torch.Tensor]:
		if roi.ndim != 4 or roi.shape[1:] != (INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
			raise ValueError(
				f"Expected ROI [B,3,{IMAGE_SIZE},{IMAGE_SIZE}], got {tuple(roi.shape)}"
			)
		cnn_features = self.backbone(roi)
		cnn_tokens = feature_map_to_tokens(cnn_features)
		projected_tokens, contextual_tokens = self.transformer(cnn_tokens)
		return {
			"cnn_features": cnn_features,
			"cnn_tokens": cnn_tokens,
			"projected_tokens": projected_tokens,
			"transformer_tokens": contextual_tokens,
		}


def print_and_validate_architecture(model: PatchCvT) -> None:
	print("=" * 60)
	print("PatchCvT Stage 1 ARCHITECTURE")
	print("=" * 60)
	print("ROI -> ResNet18 stem/layer1/layer2 -> [B,C,28,28]")
	print("-> [B,784,C] spatial tokens -> projection + positional encoding")
	print("-> Transformer -> [B,784,D] -> contextual memory bank")
	print("-> squared nearest-neighbour distance -> ROI max distance")
	print("Input ROI:             [B, 3, 224, 224]")
	print("ResNet18 feature map:  [B, 128, 28, 28]")
	print("CNN token sequence:    [B, 784, 128]")
	print(f"Transformer input:     [B, 784, {TRANSFORMER_DIM}]")
	print(f"Transformer output:    [B, 784, {TRANSFORMER_DIM}]")
	print(f"Memory bank:           [K, {TRANSFORMER_DIM}], K <= {MEMORY_BANK_SIZE}")
	print("Patch anomaly scores:  [B, 784]")
	print("ROI anomaly distance:  [B]")

	previous_mode = model.training
	model.eval()
	with torch.no_grad():
		dummy = torch.zeros((1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
		outputs = model(dummy)
	shapes = {name: tuple(value.shape) for name, value in outputs.items()}
	print("[SHAPES] input:               (1, 3, 224, 224)")
	print(f"[SHAPES] CNN feature map:     {shapes['cnn_features']}")
	print(f"[SHAPES] CNN tokens:          {shapes['cnn_tokens']}")
	print(f"[SHAPES] projected tokens:    {shapes['projected_tokens']}")
	print(f"[SHAPES] Transformer tokens:  {shapes['transformer_tokens']}")
	assert shapes["cnn_features"][-2:] == (28, 28)
	assert shapes["cnn_tokens"][1:] == (784, CNN_CHANNELS)
	assert shapes["projected_tokens"][1:] == (784, TRANSFORMER_DIM)
	assert shapes["transformer_tokens"][1:] == (784, TRANSFORMER_DIM)
	model.train(previous_mode)
	print("[SHAPES] 784-token invariant: PASS")


################################################################################
# TRAINING OBJECTIVE
################################################################################

def semantic_consistency_loss(
	clean_tokens: torch.Tensor,
	degraded_tokens: torch.Tensor,
) -> torch.Tensor:
	"""Archived semantic cosine objective; pooling occurs only inside the loss."""
	if clean_tokens.shape != degraded_tokens.shape:
		raise ValueError("Clean/degraded contextual token shapes must match")
	if clean_tokens.shape[1:] != (NUM_SPATIAL_TOKENS, TRANSFORMER_DIM):
		raise RuntimeError(f"Unexpected contextual tokens: {tuple(clean_tokens.shape)}")
	clean_roi_embedding = clean_tokens.mean(dim=1)
	degraded_roi_embedding = degraded_tokens.mean(dim=1)
	return (1.0 - F.cosine_similarity(clean_roi_embedding, degraded_roi_embedding, dim=-1)).mean()


def run_epoch(
	model: PatchCvT,
	loader: DataLoader,
	optimizer: Optional[torch.optim.Optimizer],
) -> float:
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	batch_count = 0
	for batch in loader:
		clean = batch["clean"].to(DEVICE, non_blocking=True)
		degraded = batch["degraded"].to(DEVICE, non_blocking=True)
		if training:
			optimizer.zero_grad(set_to_none=True)
			clean_tokens = model(clean)["transformer_tokens"]
			degraded_tokens = model(degraded)["transformer_tokens"]
			loss = semantic_consistency_loss(clean_tokens, degraded_tokens)
			loss.backward()
			optimizer.step()
		else:
			with torch.no_grad():
				clean_tokens = model(clean)["transformer_tokens"]
				degraded_tokens = model(degraded)["transformer_tokens"]
				loss = semantic_consistency_loss(clean_tokens, degraded_tokens)
		total_loss += float(loss.detach().item())
		batch_count += 1
	if batch_count == 0:
		raise RuntimeError("Training/validation loader has no batches")
	return total_loss / batch_count


################################################################################
# MEMORY BANK
################################################################################

@torch.no_grad()
def build_contextual_memory_bank(
	model: PatchCvT,
	loader: DataLoader,
	memory_size: int = MEMORY_BANK_SIZE,
) -> torch.Tensor:
	if memory_size <= 0:
		raise ValueError("MEMORY_BANK_SIZE must be positive")
	model.eval()
	reservoir = torch.empty((0, TRANSFORMER_DIM), dtype=torch.float32)
	reservoir_keys = torch.empty((0,), dtype=torch.float64)
	generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
	total_tokens = 0

	for batch in loader:
		images = batch["image"].to(DEVICE, non_blocking=True)
		tokens = model(images)["transformer_tokens"].detach()
		if tokens.shape[1:] != (NUM_SPATIAL_TOKENS, TRANSFORMER_DIM):
			raise RuntimeError(f"Invalid memory features: {tuple(tokens.shape)}")
		candidates = tokens.reshape(-1, TRANSFORMER_DIM).float().cpu()
		keys = torch.rand(candidates.shape[0], generator=generator, dtype=torch.float64)
		merged_features = torch.cat((reservoir, candidates), dim=0)
		merged_keys = torch.cat((reservoir_keys, keys), dim=0)
		keep_count = min(memory_size, merged_keys.numel())
		selected = torch.topk(merged_keys, k=keep_count, largest=False).indices
		reservoir = merged_features[selected].contiguous()
		reservoir_keys = merged_keys[selected].contiguous()
		total_tokens += candidates.shape[0]

	if reservoir.shape != (min(memory_size, total_tokens), TRANSFORMER_DIM):
		raise RuntimeError(f"Unexpected memory bank shape: {tuple(reservoir.shape)}")
	print(
		f"[MEMORY] contextual tokens seen={total_tokens}; selected K={reservoir.shape[0]} "
		f"uniformly by random-key reservoir sampling"
	)
	print(f"[MEMORY] bank shape={tuple(reservoir.shape)}, embedding dimension={TRANSFORMER_DIM}")
	return reservoir


def save_memory_bank(path: str, memory_bank: torch.Tensor, config: Dict[str, Any]) -> None:
	if memory_bank.ndim != 2 or memory_bank.shape[1] != TRANSFORMER_DIM:
		raise ValueError(f"Memory bank must have shape [K,{TRANSFORMER_DIM}]")
	torch.save({"memory_bank": memory_bank.cpu(), "config": config}, path)


def load_memory_bank(path: str, map_location: Any = "cpu") -> torch.Tensor:
	payload = torch.load(path, map_location=map_location, weights_only=False)
	memory_bank = payload["memory_bank"]
	if not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 2:
		raise ValueError(f"Invalid serialized memory bank at {path}")
	if memory_bank.shape[1] != TRANSFORMER_DIM:
		raise ValueError(
			f"Memory embedding dimension {memory_bank.shape[1]} != {TRANSFORMER_DIM}"
		)
	return memory_bank


################################################################################
# DISTANCE / ROI SCORING
################################################################################

@torch.no_grad()
def nearest_neighbor_patch_scores(
	contextual_tokens: torch.Tensor,
	memory_bank: torch.Tensor,
) -> torch.Tensor:
	if contextual_tokens.ndim != 3 or contextual_tokens.shape[1:] != (
		NUM_SPATIAL_TOKENS, TRANSFORMER_DIM
	):
		raise ValueError(f"Expected contextual tokens [B,784,D], got {tuple(contextual_tokens.shape)}")
	if memory_bank.ndim != 2 or memory_bank.shape[1] != TRANSFORMER_DIM:
		raise ValueError(f"Expected memory bank [K,{TRANSFORMER_DIM}], got {tuple(memory_bank.shape)}")
	if memory_bank.shape[0] == 0:
		raise ValueError("Cannot score against an empty memory bank")
	flat_tokens = contextual_tokens.reshape(-1, TRANSFORMER_DIM)
	distances = torch.cdist(flat_tokens.float(), memory_bank.float(), p=2).square()
	patch_scores = distances.min(dim=1).values.reshape(contextual_tokens.shape[0], NUM_SPATIAL_TOKENS)
	assert patch_scores.shape == (contextual_tokens.shape[0], NUM_SPATIAL_TOKENS)
	return patch_scores


def aggregate_patch_scores(
	patch_scores: torch.Tensor,
	mode: str = AGGREGATION_MODE,
	top_k: int = TOP_K_PATCHES,
) -> torch.Tensor:
	if patch_scores.ndim != 2 or patch_scores.shape[1] != NUM_SPATIAL_TOKENS:
		raise ValueError(f"Expected patch scores [B,784], got {tuple(patch_scores.shape)}")
	if mode == "max":
		return patch_scores.max(dim=1).values
	if mode == "mean":
		return patch_scores.mean(dim=1)
	if mode == "topk":
		if not 1 <= top_k <= NUM_SPATIAL_TOKENS:
			raise ValueError(f"top_k must be between 1 and {NUM_SPATIAL_TOKENS}")
		return patch_scores.topk(top_k, dim=1).values.mean(dim=1)
	raise ValueError(f"Unknown aggregation mode: {mode}")


def distance_to_viability(raw_roi_distance: torch.Tensor) -> torch.Tensor:
	if DISTANCE_TO_VIABILITY_SCALE <= 0:
		raise ValueError("DISTANCE_TO_VIABILITY_SCALE must be positive")
	viability = torch.exp(-raw_roi_distance.clamp_min(0) / DISTANCE_TO_VIABILITY_SCALE)
	return viability.clamp(0.0, 1.0)


################################################################################
# CHECKPOINTING / TRAINING CONFIGURATION
################################################################################

def configuration_dict() -> Dict[str, Any]:
	return {
		"project_root": str(PROJECT_ROOT),
		"train_dataset": str(TRAIN_DATASET),
		"test_dataset": str(TEST_DATASET),
		"memory_bank_dataset": str(MEMORY_BANK_DATASET),
		"image_size": IMAGE_SIZE,
		"backbone": "ResNet18 stem + layer1 + layer2 (frozen)",
		"pretrained_resnet18": USE_PRETRAINED_RESNET18,
		"cnn_channels": CNN_CHANNELS,
		"feature_map_size": FEATURE_MAP_SIZE,
		"spatial_tokens": NUM_SPATIAL_TOKENS,
		"transformer_dim": TRANSFORMER_DIM,
		"transformer_depth": TRANSFORMER_DEPTH,
		"attention_heads": NUM_ATTENTION_HEADS,
		"dropout": TRANSFORMER_DROPOUT,
		"feedforward_dim": TRANSFORMER_FEEDFORWARD_DIM,
		"epochs": EPOCHS,
		"batch_size": BATCH_SIZE,
		"learning_rate": LEARNING_RATE,
		"weight_decay": WEIGHT_DECAY,
		"optimizer": "AdamW",
		"scheduler": None,
		"training_objective": "1 - cosine_similarity(mean contextual clean tokens, mean contextual degraded tokens)",
		"memory_bank_size_cap": MEMORY_BANK_SIZE,
		"memory_bank_selection": "uniform random-key reservoir sample from all contextual memorybank tokens",
		"aggregation": AGGREGATION_MODE,
		"distance_to_viability": "exp(-raw_squared_nn_distance / DISTANCE_TO_VIABILITY_SCALE)",
		"distance_to_viability_scale": DISTANCE_TO_VIABILITY_SCALE,
		"random_seed": RANDOM_SEED,
	}


def save_checkpoint(
	path: str,
	model: PatchCvT,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	validation_loss: float,
	memory_bank: Optional[torch.Tensor] = None,
	optimizer_state: Optional[Dict[str, Any]] = None,
) -> None:
	checkpoint = {
		"epoch": epoch,
		"validation_loss": validation_loss,
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict() if optimizer_state is None else optimizer_state,
		"config": configuration_dict(),
		"memory_bank": None if memory_bank is None else memory_bank.cpu(),
	}
	torch.save(checkpoint, path)


def load_model_from_checkpoint(path: str) -> Tuple[PatchCvT, Dict[str, Any]]:
	checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
	checkpoint_config = checkpoint["config"]
	if checkpoint_config["transformer_dim"] != TRANSFORMER_DIM:
		raise ValueError("Checkpoint transformer configuration differs from this script")
	model = PatchCvT(pretrained_resnet18=False).to(DEVICE)
	model.load_state_dict(checkpoint["model_state"])
	return model, checkpoint


################################################################################
# TRAINING LOOP
################################################################################

def train_model() -> Tuple[PatchCvT, torch.Tensor]:
	set_random_seed(RANDOM_SEED)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)

	train_samples = collect_roi_samples(TRAIN_DATASET)
	memory_samples = collect_roi_samples(MEMORY_BANK_DATASET)
	test_samples = collect_roi_samples(TEST_DATASET)
	if len(train_samples) < 2:
		raise RuntimeError("Need at least two annotated training ROIs for a train/validation split")
	if not memory_samples:
		raise RuntimeError("memorybank/ contains no annotated reference ROIs")
	if not test_samples:
		raise RuntimeError("database/test contains no annotated ROIs")

	shuffled_indices = list(range(len(train_samples)))
	random.Random(RANDOM_SEED).shuffle(shuffled_indices)
	validation_count = max(1, int(round(len(train_samples) * TRAIN_VALIDATION_FRACTION)))
	validation_indices = shuffled_indices[:validation_count]
	training_indices = shuffled_indices[validation_count:]
	if not training_indices:
		raise RuntimeError("Train/validation split left no training ROIs")
	training_subset = [train_samples[index] for index in training_indices]
	validation_subset = [train_samples[index] for index in validation_indices]

	eval_transform = build_eval_transform()
	train_dataset = ROIPairDataset(
		training_subset,
		eval_transform,
		degradation_transform=build_degradation_transform(),
	)
	validation_dataset = ROIPairDataset(
		validation_subset,
		eval_transform,
		degradation_transform=build_validation_degradation(),
	)
	train_loader = build_loader(train_dataset, BATCH_SIZE, shuffle=True)
	validation_loader = build_loader(validation_dataset, BATCH_SIZE, shuffle=False)
	memory_loader = build_loader(
		ROIPairDataset(memory_samples, eval_transform), MEMORY_BATCH_SIZE, shuffle=False
	)

	model = PatchCvT().to(DEVICE)
	print_and_validate_architecture(model)
	trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	frozen_parameters = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
	trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
	print(f"[PARAMS] trainable={trainable_count:,}; frozen={frozen_parameters:,}")
	print("[TRAIN] Trainable: CNN-to-D projection, learned 784-position embedding, Transformer.")
	print("[TRAIN] Frozen: ResNet18 stem/layer1/layer2. Objective: archived paired semantic cosine loss.")
	print("[TRAIN] No classification labels, heads, or test-set checkpoint selection are used.")

	optimizer = torch.optim.AdamW(
		trainable_parameters,
		lr=LEARNING_RATE,
		weight_decay=WEIGHT_DECAY,
	)
	best_validation_loss = float("inf")
	best_epoch = -1
	best_state: Optional[Dict[str, torch.Tensor]] = None
	best_optimizer_state: Optional[Dict[str, Any]] = None

	for epoch in range(1, EPOCHS + 1):
		started = time.perf_counter()
		train_loss = run_epoch(model, train_loader, optimizer)
		validation_loss = run_epoch(model, validation_loader, None)
		elapsed = time.perf_counter() - started
		learning_rate = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch:02d}/{EPOCHS} | train loss={train_loss:.6f} "
			f"| validation loss={validation_loss:.6f} | lr={learning_rate:.2e} "
			f"| duration={elapsed:.1f}s"
		)
		save_checkpoint(LAST_CHECKPOINT_PATH, model, optimizer, epoch, validation_loss)
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			best_epoch = epoch
			best_state = copy.deepcopy(model.state_dict())
			best_optimizer_state = copy.deepcopy(optimizer.state_dict())
			save_checkpoint(BEST_CHECKPOINT_PATH, model, optimizer, epoch, validation_loss)
			print(f"[CHECKPOINT] best.pt updated at epoch {epoch}")

	if best_state is None or best_optimizer_state is None:
		raise RuntimeError("Training completed without a best checkpoint")

	# The memory bank is constructed only after the selected encoder is fixed.
	model.load_state_dict(best_state)
	best_memory_bank = build_contextual_memory_bank(model, memory_loader)
	if best_memory_bank.shape[1] != TRANSFORMER_DIM:
		raise RuntimeError("Contextual memory-bank dimensionality invariant failed")
	save_memory_bank(MEMORY_BANK_PATH, best_memory_bank, configuration_dict())
	save_checkpoint(
		BEST_CHECKPOINT_PATH,
		model,
		optimizer,
		best_epoch,
		best_validation_loss,
		best_memory_bank,
		best_optimizer_state,
	)

	# Store a matching bank in last.pt too, so either checkpoint can reproduce inference.
	last_model, last_checkpoint = load_model_from_checkpoint(LAST_CHECKPOINT_PATH)
	last_memory_bank = build_contextual_memory_bank(last_model, memory_loader)
	save_checkpoint(
		LAST_CHECKPOINT_PATH,
		last_model,
		optimizer,
		int(last_checkpoint["epoch"]),
		float(last_checkpoint["validation_loss"]),
		last_memory_bank,
	)
	print(f"[CHECKPOINT] best={BEST_CHECKPOINT_PATH}; last={LAST_CHECKPOINT_PATH}")
	return model, best_memory_bank.to(DEVICE)


################################################################################
# STANDALONE EVALUATION
################################################################################

@torch.no_grad()
def evaluate_test_set(model: PatchCvT, memory_bank: torch.Tensor) -> List[Dict[str, Any]]:
	test_samples = collect_roi_samples(TEST_DATASET)
	test_loader = build_loader(
		ROIPairDataset(test_samples, build_eval_transform()),
		EVAL_BATCH_SIZE,
		shuffle=False,
	)
	model.eval()
	memory_bank = memory_bank.to(DEVICE)
	results: List[Dict[str, Any]] = []

	print("\n[TEST] Continuous nearest-neighbour distance evaluation")
	for batch in test_loader:
		images = batch["image"].to(DEVICE, non_blocking=True)
		contextual_tokens = model(images)["transformer_tokens"]
		patch_scores = nearest_neighbor_patch_scores(contextual_tokens, memory_bank)
		max_distance = aggregate_patch_scores(patch_scores, "max")
		mean_distance = aggregate_patch_scores(patch_scores, "mean")
		topk_distance = aggregate_patch_scores(patch_scores, "topk", TOP_K_PATCHES)
		viability = distance_to_viability(max_distance)

		for batch_index in range(images.shape[0]):
			class_id = int(batch["class_id"][batch_index])
			row: Dict[str, Any] = {
				"image": batch["image_path"][batch_index],
				"object_index": int(batch["object_index"][batch_index]),
				"gt_class_id": class_id,
				"roi_nn_distance_max": float(max_distance[batch_index].item()),
				"roi_nn_distance_mean": float(mean_distance[batch_index].item()),
				"roi_nn_distance_topk": float(topk_distance[batch_index].item()),
				"viability_score": float(viability[batch_index].item()),
			}
			if VALIDITY_THRESHOLD is None:
				predicted_validity: Any = "not_thresholded"
			else:
				predicted_validity = bool(viability[batch_index].item() >= VALIDITY_THRESHOLD)
			row["predicted_validity"] = predicted_validity
			results.append(row)
			print(
				f"image={row['image']} object={row['object_index']} gt_class_id={class_id} "
				f"nn_max={row['roi_nn_distance_max']:.6f} "
				f"nn_mean={row['roi_nn_distance_mean']:.6f} "
				f"nn_top{TOP_K_PATCHES}={row['roi_nn_distance_topk']:.6f} "
				f"viability={row['viability_score']:.6f} "
				f"predicted_validity={predicted_validity}"
			)

	if not results:
		raise RuntimeError("No test ROIs were evaluated")
	with open(EVALUATION_CSV_PATH, "w", newline="", encoding="utf-8") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=list(results[0].keys()))
		writer.writeheader()
		writer.writerows(results)

	max_distances = np.asarray([row["roi_nn_distance_max"] for row in results])
	viability_scores = np.asarray([row["viability_score"] for row in results])
	print(
		f"[TEST SUMMARY] ROIs={len(results)}; max-distance mean={max_distances.mean():.6f}, "
		f"std={max_distances.std():.6f}; viability mean={viability_scores.mean():.6f}"
	)
	print("[TEST SUMMARY] YOLO class IDs are reported as metadata, not treated as validity labels.")
	print(f"[TEST] Per-ROI results saved to {EVALUATION_CSV_PATH}")
	return results


################################################################################
# MAIN
################################################################################

def main() -> None:
	print(f"Device: {DEVICE}")
	print(f"Pretrained ResNet18: {USE_PRETRAINED_RESNET18}")
	print(f"ROI viability mapping: exp(-distance / {DISTANCE_TO_VIABILITY_SCALE})")
	model, memory_bank = train_model()
	assert memory_bank.ndim == 2 and memory_bank.shape[1] == TRANSFORMER_DIM
	print(f"[SHAPES] memory bank dimensionality: {tuple(memory_bank.shape)}; PASS")
	evaluate_test_set(model, memory_bank)


if __name__ == "__main__":
	main()
