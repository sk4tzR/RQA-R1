# modeling_rqa.py

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
)

# ============================================================
# CONFIG
# ============================================================

class RQAModelConfig(PretrainedConfig):
    model_type = "rqa"

    def __init__(
        self,
        base_model_name: str = "FacebookAI/xlm-roberta-large",
        num_error_types: int = 6,
        has_issue_projection_dim: int = 256,
        errors_projection_dim: int = 512,
        has_issue_dropout: float = 0.25,
        errors_dropout: float = 0.3,
        temperature_has_issue: float = 1.0,
        temperature_errors: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.num_error_types = num_error_types
        self.has_issue_projection_dim = has_issue_projection_dim
        self.errors_projection_dim = errors_projection_dim
        self.has_issue_dropout = has_issue_dropout
        self.errors_dropout = errors_dropout

        self.temperature_has_issue = temperature_has_issue
        self.temperature_errors = (
            temperature_errors
            if temperature_errors is not None
            else [1.0] * num_error_types
        )

# ============================================================
# POOLING
# ============================================================

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

# ============================================================
# MODEL
# ============================================================

class RQAModelHF(PreTrainedModel):
    config_class = RQAModelConfig

    def __init__(self, config: RQAModelConfig):
        super().__init__(config)

        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.encoder.config.hidden_size

        self.pooler = MeanPooling()

        self.has_issue_projection = nn.Sequential(
            nn.Linear(hidden_size, config.has_issue_projection_dim),
            nn.LayerNorm(config.has_issue_projection_dim),
            nn.GELU(),
            nn.Dropout(config.has_issue_dropout),
        )

        self.errors_projection = nn.Sequential(
            nn.Linear(hidden_size, config.errors_projection_dim),
            nn.LayerNorm(config.errors_projection_dim),
            nn.GELU(),
            nn.Dropout(config.errors_dropout),
        )

        self.has_issue_head = nn.Linear(config.has_issue_projection_dim, 1)
        self.errors_head = nn.Linear(
            config.errors_projection_dim, config.num_error_types
        )

        self._init_custom_weights()

    def _init_custom_weights(self):
        for module in [
            self.has_issue_projection[0],
            self.errors_projection[0],
            self.has_issue_head,
            self.errors_head,
        ]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled = self.pooler(outputs.last_hidden_state, attention_mask)

        has_issue_logits = self.has_issue_head(
            self.has_issue_projection(pooled)
        ).squeeze(-1)

        errors_logits = self.errors_head(
            self.errors_projection(pooled)
        )

        return {
            "has_issue_logits": has_issue_logits,
            "errors_logits": errors_logits,
        }

# ============================================================
# 🔥 TRANSFORMERS REGISTRATION (КРИТИЧНО)
# ============================================================

AutoConfig.register("rqa", RQAModelConfig)
AutoModel.register(RQAModelConfig, RQAModelHF)

print("✅ RQA зарегистрирован в Transformers")
