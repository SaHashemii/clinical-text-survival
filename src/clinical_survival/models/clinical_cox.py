"""
Clinical unimodal Cox model.
"""

from __future__ import annotations

import torch
from torch import nn

from clinical_survival.models.encoders.clinical import (
    ClinicalEmbeddingEncoder,
    ClinicalEmbeddingPoolingEncoder,
    ClinicalEncoder,
)
from clinical_survival.models.heads import CoxHead


class ClinicalCoxModel(nn.Module):
    """Clinical-only Cox model for tabular covariates or precomputed embeddings."""

    def __init__(
        self,
        clinical_source: str,
        clinical_dim: int,
        clinical_token_count: int | None = None,
        clinical_hidden_dims: list[int] | None = None,
        clinical_emb_dim: int = 128,
        clinical_token_hidden_dim: int = 256,
        clinical_token_out_dim: int = 128,
        clinical_dropout: float = 0.30,
        clinical_activation: str = "selu",
        clinical_pooling: str | None = None,
        clinical_projection_dim: int = 512,
        clinical_attention_hidden_dim: int = 128,
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.30,
        head_activation: str = "selu",
    ):
        super().__init__()
        clinical_hidden_dims = clinical_hidden_dims or [128]
        if clinical_source == "embedding":
            if clinical_token_count is None:
                raise ValueError("clinical_token_count is required when clinical_source='embedding'.")
            if clinical_pooling is None:
                self.encoder = ClinicalEmbeddingEncoder(
                    token_count=clinical_token_count,
                    in_dim=clinical_dim,
                    token_hidden_dim=clinical_token_hidden_dim,
                    token_out_dim=clinical_token_out_dim,
                    emb_dim=clinical_emb_dim,
                    dropout=clinical_dropout,
                    activation=clinical_activation,
                )
                head_in_dim = clinical_emb_dim
            else:
                self.encoder = ClinicalEmbeddingPoolingEncoder(
                    token_count=clinical_token_count,
                    in_dim=clinical_dim,
                    pooling=clinical_pooling,
                    attention_hidden_dim=clinical_attention_hidden_dim,
                    projection_dim=clinical_projection_dim,
                    dropout=clinical_dropout,
                )
                head_in_dim = self.encoder.output_dim
        elif clinical_source == "tabular":
            self.encoder = ClinicalEncoder(
                in_dim=clinical_dim,
                hidden_dims=clinical_hidden_dims,
                emb_dim=clinical_emb_dim,
                dropout=clinical_dropout,
                activation=clinical_activation,
            )
            head_in_dim = clinical_emb_dim
        else:
            raise ValueError(f"Unknown clinical_source: {clinical_source}")

        self.head = CoxHead(
            in_dim=head_in_dim,
            hidden_dims=head_hidden_dims or [],
            dropout=head_dropout,
            activation=head_activation,
        )

    def forward_all(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(clinical))

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.forward_all(clinical)
