"""Factories for training/evaluation loss objects."""

from __future__ import annotations

from typing import Tuple

from src.config import PhysicsConfig
from src.losses.physics import EnhancedPhysicsLoss


def create_physics_loss(
    physics_cfg: PhysicsConfig,
    *,
    enzyme_confidence: float,
    flux_flexibility: float,
) -> Tuple[object, bool]:
    """Return the physics loss instance and whether listwise ranking is enabled."""
    listwise_config = {
        'enzyme_confidence': enzyme_confidence,
        'flux_flexibility': flux_flexibility,
        'min_reactions': 1,
        'use_absolute_flux': True,
        'flux_sign_weighting': False,
    }

    effective_use_listwise = physics_cfg.use_listwise_ranking or (physics_cfg.lambda_kl > 0)
    if physics_cfg.lambda_kl > 0 and not physics_cfg.use_listwise_ranking:
        print(f"INFO: Automatically enabling listwise ranking since λ_kl={physics_cfg.lambda_kl} > 0")

    loss = EnhancedPhysicsLoss(
        λ_mse=1.0,
        λ_bal=physics_cfg.lambda_balance,
        λ_corr=physics_cfg.lambda_correlation,
        λ_var=physics_cfg.lambda_variance,
        λ_kl=physics_cfg.lambda_kl,
        use_listwise_ranking=effective_use_listwise,
        listwise_config=listwise_config,
    )

    return loss, effective_use_listwise
