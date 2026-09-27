"""Etapa 4: un Transformer decoder-only (estilo GPT) escrito desde cero.

Recorrido de los datos:

    ids (B, T)                     enteros: un token por posición
      -> wte + wpe  (B, T, C)      embedding del token + embedding de su posición
      -> N x Block  (B, T, C)      cada bloque: atención causal + MLP (con residuales)
      -> ln_f       (B, T, C)      normalización final
      -> lm_head    (B, T, V)      una puntuación (logit) por cada token del vocabulario

B = batch, T = posiciones (<= block_size), C = n_embd, V = vocab_size.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.config import Config


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False

    @classmethod
    def from_config(cls, cfg: Config, vocab_size: int | None = None) -> GPTConfig:
        return cls(
            vocab_size=vocab_size or cfg.data.vocab_size,
            block_size=cfg.data.block_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            dropout=cfg.model.dropout,
            bias=cfg.model.bias,
        )


class CausalSelfAttention(nn.Module):
    """Atención multi-cabeza causal: cada token "mira" a los anteriores, nunca a los futuros.

    Para cada token se calculan tres vectores:
      q (query) "qué busco",  k (key) "qué ofrezco",  v (value) "qué información llevo".
    La afinidad entre tokens es q·k; con softmax se convierte en pesos que mezclan los v.
    Varias cabezas permiten atender a relaciones distintas en paralelo.
    """

    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        if c.n_embd % c.n_head != 0:
            raise ValueError("n_embd debe ser divisible por n_head")
        self.n_head = c.n_head
        self.head_dim = c.n_embd // c.n_head
        self.dropout = c.dropout
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)  # q, k, v de una vez
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.resid_dropout = nn.Dropout(c.dropout)
        self.use_sdpa = True  # False = implementación manual (más lenta, más didáctica)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): cada cabeza trabaja por separado
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            # Versión optimizada de PyTorch; matemáticamente igual a manual_attention
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            y = self.manual_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # juntar las cabezas
        return self.resid_dropout(self.c_proj(y))

    def manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """softmax(q·kᵀ / √d + máscara causal) · v, paso a paso."""
        T = q.size(-2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (B, nh, T, T)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        att = att.masked_fill(future, float("-inf"))                         # prohibido mirar al futuro
        att = F.softmax(att, dim=-1)                                         # cada fila suma 1
        att = F.dropout(att, p=self.dropout, training=self.training)
        return att @ v                                                       # (B, nh, T, head_dim)


class MLP(nn.Module):
    """Red feed-forward por posición: expande a 4·C, aplica GELU y vuelve a C."""

    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Bloque Transformer (pre-norm): x + Atención(LN(x)), luego x + MLP(LN(x)).

    Las conexiones residuales (el "x +") dejan pasar la información y el gradiente
    directamente, lo que permite apilar muchos bloques sin que el entrenamiento se rompa.
    """

    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.attn = CausalSelfAttention(c)
        self.ln_2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = MLP(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # comunicación entre tokens
        x = x + self.mlp(self.ln_2(x))   # "pensar" en cada token por separado
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token -> vector
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # posición -> vector
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: la matriz que convierte token->vector se reutiliza para vector->token.
        # Ahorra V·C parámetros (~1 M aquí) y suele mejorar los resultados.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Las proyecciones que suman al residual se inicializan más pequeñas (GPT-2)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Devuelve (logits, loss). `loss` es la entropía cruzada si se pasan `targets`."""
        _, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Secuencia de {T} tokens > block_size={self.config.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            # reshape (no view): funciona aunque los tensores no sean contiguos en memoria
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self) -> dict[str, int]:
        """Parámetros por componente (el lm_head comparte pesos con wte y no suma)."""
        blocks = sum(p.numel() for p in self.blocks.parameters())
        return {
            "embeddings_tokens": self.wte.weight.numel(),
            "embeddings_posicion": self.wpe.weight.numel(),
            "bloques_transformer": blocks,
            "norm_final": sum(p.numel() for p in self.ln_f.parameters()),
            "total": sum(p.numel() for p in self.parameters()),  # parameters() no duplica
        }

    def generate_iter(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eot_id: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Genera tokens uno a uno (autorregresivo) y entrega cada uno en cuanto existe.

        temperature: <1 más conservador, >1 más creativo, 0 = siempre el más probable.
        top_k: solo se sortea entre los k tokens más probables.
        """
        for _ in range(max_new_tokens):
            with torch.no_grad():
                idx_cond = idx[:, -self.config.block_size :]  # el modelo solo ve block_size tokens
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]                      # solo importa la última posición
                if temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield next_id
            if eot_id is not None and bool((next_id == eot_id).all()):
                return

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eot_id: int | None = None,
    ) -> torch.Tensor:
        """Devuelve el contexto `idx` (B, T) seguido de los tokens generados."""
        new = list(self.generate_iter(idx, max_new_tokens, temperature, top_k, eot_id))
        return torch.cat([idx, *new], dim=1)
