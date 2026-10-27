"""PatchCvT memory-conditioned tokeniser + modulated DETR (PatchDETR).

Equation numbers refer to patch-cv-detr.pdf (sec. 11-12) / PatchCvT_Extended_Formulation.pdf.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .config import PatchDETRConfig


class ResNetBackbone(nn.Module):
	"""f_theta (eq. 2/31): ImageNet ResNet18 stem..layer2 [..layer3]. BatchNorm is always in
	eval mode. Everything is frozen (as in patchCvT/train.py) unless `train_last` unfreezes
	layer3, which is the only way L_memory (eq. 17) can receive a gradient."""

	def __init__(self, pretrained: bool = True, use_layer3: bool = True, train_last: bool = False) -> None:
		super().__init__()
		self.pretrained_loaded = False
		try:
			net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
			self.pretrained_loaded = pretrained
		except Exception as exc:  # offline
			print(f"[WARN] could not load pretrained ResNet18 ({exc}); using random init")
			net = resnet18(weights=None)
		self.frozen = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
		self.tail = net.layer3 if use_layer3 else nn.Identity()
		self.out_channels = 256 if use_layer3 else 128
		for p in self.parameters():
			p.requires_grad = False
		self.trainable_tail = bool(train_last and use_layer3)
		if self.trainable_tail:
			for p in self.tail.parameters():
				p.requires_grad = True
		self.eval()

	def train(self, mode: bool = True) -> "ResNetBackbone":
		return super().train(False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			x = self.frozen(x)
		with torch.set_grad_enabled(self.trainable_tail and torch.is_grad_enabled()):
			return self.tail(x)


def sine_position_embedding(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
	"""2D sine positional encoding -> [1, h*w, dim]."""
	d = dim // 4
	omega = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=torch.float32) / d))
	ys, xs = torch.meshgrid(torch.arange(h, device=device).float(), torch.arange(w, device=device).float(), indexing="ij")
	oy, ox = ys.reshape(-1, 1) * omega, xs.reshape(-1, 1) * omega
	return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)[None]


class PatchCvTModule(nn.Module):
	"""Eq. 21-30: soft memory retrieval, residual/energy tokens, transformer, robustness
	head R and semantic-preservation gate S."""

	def __init__(self, in_channels: int, dim: int, depth: int, heads: int, tau: float, dropout: float) -> None:
		super().__init__()
		self.dim, self.tau = dim, tau
		self.register_buffer("memory", torch.zeros(0, in_channels))  # M: buffer => never optimised (eq. 18)
		self.proj = nn.Linear(3 * in_channels + 1, dim)  # z_j = [q || q~ || r || a]
		layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
		self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
		self.robust_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))  # g_phi
		self.w_r = nn.Parameter(torch.tensor(4.0))
		self.w_e = nn.Parameter(torch.tensor(1.0))

	def set_memory(self, memory: torch.Tensor) -> None:
		if memory.ndim != 2 or memory.shape[1] != self.memory.shape[1] or memory.shape[0] == 0:
			raise ValueError(f"memory must be [K,{self.memory.shape[1]}], got {tuple(memory.shape)}")
		self.memory = memory.detach().to(self.memory.device, torch.float32).contiguous()

	def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
		key = prefix + "memory"
		if key in state_dict:  # K is data dependent; adopt the checkpoint's bank size
			self.memory = torch.zeros_like(state_dict[key])
		super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

	def retrieve(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Eq. 21-24. Returns (q~, r, alpha). Gradients flow to q (and through alpha), never to M."""
		if self.memory.shape[0] == 0:
			raise RuntimeError("memory bank is empty; call set_memory() first")
		with torch.autocast(q.device.type, enabled=False):
			q = q.float()
			M = self.memory
			d = torch.cdist(q, M.unsqueeze(0).expand(q.shape[0], -1, -1))  # d_jk (eq. 21)
			alpha = torch.softmax(-d / self.tau, dim=-1)  # eq. 22
			q_tilde = alpha @ M  # eq. 23
		return q_tilde, q - q_tilde, alpha

	def forward(self, feat: torch.Tensor) -> Dict[str, Any]:
		B, C, h, w = feat.shape
		F_tok = feat.float().flatten(2).transpose(1, 2)  # [B,N,C]
		a = F_tok.norm(dim=-1, keepdim=True)  # eq. 10 activation energy (pre-normalisation)
		q = F.normalize(F_tok, dim=-1)  # q_j, unit norm so that ||r|| is in [0,2]
		q_tilde, r, _ = self.retrieve(q)
		E = r.norm(dim=-1)  # eq. 25 (= ||r_j|| used in eq. 29 / 14)
		z = torch.cat([q, q_tilde, r, a], dim=-1)  # eq. 26
		x = self.proj(z) + sine_position_embedding(h, w, self.dim, feat.device)
		z_prime = self.encoder(x)  # eq. 27
		R = torch.sigmoid(self.robust_head(z_prime).squeeze(-1).float())  # eq. 28, in (0,1)
		S = torch.sigmoid(self.w_r * R - self.w_e * E)  # eq. 29
		z_hat = S.unsqueeze(-1).to(z_prime.dtype) * z_prime  # eq. 30
		return {"z_hat": z_hat, "z_prime": z_prime, "R": R, "S": S, "E": E, "r": r, "q": q, "q_tilde": q_tilde, "hw": (h, w)}


def expand_bias_mask(bias: torch.Tensor, heads: int, n_queries: int) -> torch.Tensor:
	"""[B,N] additive key bias -> [B*heads, Q, N] float mask for nn.MultiheadAttention."""
	B, N = bias.shape
	return bias[:, None, None, :].expand(B, heads, n_queries, N).reshape(B * heads, n_queries, N)


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
	"""DETR decoder layer; cross-attention logits are QK^T/sqrt(d) + bias (eq. 34)."""

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
		mask = expand_bias_mask(bias, self.heads, tgt.shape[1]).to(tgt.dtype)
		attn = self.cross_attn(tgt + query_pos, memory + mem_pos, memory, attn_mask=mask, need_weights=False)[0]
		tgt = self.n2(tgt + self.drop(attn))
		return self.n3(tgt + self.drop(self.ffn(tgt)))


class PatchDETR(nn.Module):
	def __init__(self, cfg: PatchDETRConfig) -> None:
		super().__init__()
		m = cfg.model
		self.cfg = cfg
		self.backbone = ResNetBackbone(m.pretrained, not m.stride8, m.train_backbone)
		self.patchcvt = PatchCvTModule(self.backbone.out_channels, m.dim, m.enc_layers, m.heads, m.tau, m.dropout)
		self.queries = nn.Embedding(m.num_queries, m.dim)
		self.query_pos = nn.Embedding(m.num_queries, m.dim)
		self.decoder = nn.ModuleList(ModulatedDecoderLayer(m.dim, m.heads, m.dropout) for _ in range(m.dec_layers))
		self.dec_norm = nn.LayerNorm(m.dim)
		self.class_head = nn.Linear(m.dim, 2)  # TCD + no-object
		self.box_head = MLP(m.dim, m.dim, 4, 3)  # (x, y, w, h); depth z is not supervised in this dataset
		self.heat_head = nn.Linear(m.dim, 1)  # uncertainty-aware heatmap h
		self.lam_motion = nn.Parameter(torch.tensor(1.0))  # lambda_1 (eq. 34)
		self.lam_robust = nn.Parameter(torch.tensor(1.0))  # lambda_2
		self.gamma = nn.Parameter(torch.tensor(2.0))  # eq. 33
		self.delta = nn.Parameter(torch.tensor(1.0))

	def encode(self, images: torch.Tensor) -> Dict[str, Any]:
		return self.patchcvt(self.backbone(images))

	def attention_bias(self, enc: Dict[str, Any], motion: torch.Tensor) -> Dict[str, torch.Tensor]:
		"""Eq. 32-34 additive term. P_robust uses E scaled by its (detached) per-image mean so
		gamma/delta are scale free - a documented normalisation of eq. 33."""
		h, w = enc["hw"]
		P_motion = F.adaptive_avg_pool2d(motion.float(), (h, w)).flatten(1)
		E = enc["E"]
		E_rel = E / (E.mean(dim=1, keepdim=True).detach() + 1e-6)
		P_robust = torch.sigmoid(self.gamma * (E_rel - self.delta))
		return {"bias": self.lam_motion * P_motion + self.lam_robust * P_robust, "P_motion": P_motion, "P_robust": P_robust}

	def detect(self, enc: Dict[str, Any], motion: torch.Tensor) -> Dict[str, Any]:
		z_hat, (h, w) = enc["z_hat"], enc["hw"]
		B, N, D = z_hat.shape
		prior = self.attention_bias(enc, motion)
		mem_pos = sine_position_embedding(h, w, D, z_hat.device)
		qpos = self.query_pos.weight[None].expand(B, -1, -1)
		tgt = self.queries.weight[None].expand(B, -1, -1)
		outs = []
		for layer in self.decoder:
			tgt = layer(tgt, qpos, z_hat, mem_pos, prior["bias"])
			hs = self.dec_norm(tgt)
			outs.append({"logits": self.class_head(hs), "boxes": self.box_head(hs).sigmoid()})
		heat = self.heat_head(z_hat).squeeze(-1).reshape(B, h, w)
		return {**outs[-1], "aux": outs[:-1], "heatmap": heat, **prior}

	def forward(self, images: torch.Tensor, motion: torch.Tensor) -> Dict[str, Any]:
		enc = self.encode(images)
		return {**self.detect(enc, motion), "enc": enc}
