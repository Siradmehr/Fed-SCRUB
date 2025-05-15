import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

# Consolidated epsilon constants - defined once to avoid dictionary lookups
EPS_TINY = 1e-6
EPS_SMALL = 1e-5
EPS_MEDIUM = 5e-3
EPS_LARGE = 5e-2


class BaseDivergence(nn.Module):
    """Base class for all divergence measures"""

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = True, softmax_gt: bool = True):
        """
        Initialize base divergence measure.

        Args:
            nclass: Number of classes
            param: Optional parameters for specific divergence measures
            softmax_logits: Whether to apply softmax to model predictions
            softmax_gt: Whether to apply softmax to ground truth distributions
        """
        super(BaseDivergence, self).__init__()
        self.nclass = nclass
        self.param = [] if param is None else param
        self.softmax_logits = softmax_logits
        self.softmax_gt = softmax_gt
        assert nclass >= 2, "Number of classes must be at least 2"

    def prepare_inputs(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process inputs based on their shapes.

        Args:
            logits: Model predictions
            targets: Ground truth (class indices or distributions)

        Returns:
            Processed logits and targets
        """
        if len(targets.shape) == len(logits.shape) - 1:
            # If targets are class indices, convert to one-hot
            targets = F.one_hot(targets, self.nclass).to(logits.dtype).to(logits.device)

            # If targets are already distributions
        if self.softmax_logits:
            logits = F.softmax(logits, dim=1)

        # Only apply softmax to targets if they're not already one-hot encoded and softmax_gt is True
        if self.softmax_gt and not torch.allclose(targets.sum(dim=1),
                                                  torch.ones(targets.size(0), device=targets.device)):
            targets = F.softmax(targets, dim=1)

        return logits, targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """To be implemented by child classes"""
        raise NotImplementedError


class KL(BaseDivergence):
    """Kullback-Leibler Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Using clamp for numerical stability instead of adding epsilon
        log_probs = torch.log(torch.clamp(logits, min=EPS_TINY))
        return (-targets * log_probs).sum(dim=1).mean()


class TV(BaseDivergence):
    """Total Variation Distance"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Use torch.where instead of creating a factor tensor
        diff = torch.abs(logits - targets)
        return diff.sum(dim=1).mean() * 0.5


class X2(BaseDivergence):
    """Chi-Square Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Simplified computation
        diff_squared = torch.pow(targets - logits, 2)
        loss = diff_squared / (logits + EPS_MEDIUM)
        return loss.sum(dim=1).mean() * 0.5


class PowD(BaseDivergence):
    """Power Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid alpha parameter for Power Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        C = torch.pow(logits, alpha - 1)
        loss = (torch.pow(targets, alpha) - C * logits) / (C + EPS_MEDIUM)
        return loss.sum(dim=1).mean()


class JS(BaseDivergence):
    """Jensen-Shannon Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # More numerically stable computation
        mixture = 0.5 * (logits + targets)
        log_mixture = torch.log(torch.clamp(mixture, min=EPS_TINY))
        log_logits = torch.log(torch.clamp(logits, min=EPS_TINY))
        log_targets = torch.log(torch.clamp(targets, min=EPS_TINY))

        loss = 0.5 * (
                targets * (log_targets - log_mixture) +
                logits * (log_logits - log_mixture)
        )
        return loss.sum(dim=1).mean()


class GenKL(BaseDivergence):
    """Generalized Kullback-Leibler Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid alpha parameter for Generalized KL")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        C = torch.pow(logits, alpha - 1)
        loss = ((C * logits) - torch.pow(targets, alpha)) / (alpha * (C + EPS_SMALL))
        return loss.sum(dim=1).mean()


class Exp(BaseDivergence):
    """Exponential Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid parameter for Exponential Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        # More stable computation with torch.clamp
        denom = torch.clamp(logits + EPS_LARGE * 10, min=EPS_TINY)
        exp_term = torch.exp((targets - logits) / denom) - 1
        loss = logits * exp_term
        return loss.sum(dim=1).mean()


class LeCam(BaseDivergence):
    """Le Cam Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid parameter for Le Cam Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        diff = logits - targets
        denom = logits + targets + EPS_MEDIUM
        loss = 0.5 * (diff * logits / denom)
        return loss.sum(dim=1).mean()


class AlphaRenyi(BaseDivergence):
    """Alpha-Renyi Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] < 0:
            raise ValueError("Invalid alpha parameter for Alpha-Renyi Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        # Handle alpha == 1 specially (limit case is KL divergence)
        if abs(alpha - 1.0) < EPS_TINY:
            # Return KL divergence for alpha ≈ 1
            log_ratio = torch.log(torch.clamp(targets / torch.clamp(logits, min=EPS_TINY), min=EPS_TINY))
            return (targets * log_ratio).sum(dim=1).mean()

        P = torch.pow(targets, alpha)
        Q = torch.pow(logits, 1 - alpha)
        loss = P * Q
        loss = torch.log(torch.clamp(loss.sum(dim=1), min=EPS_MEDIUM))
        return loss.mean() / (alpha - 1)


class BetaSkew(BaseDivergence):
    """Beta-Skew Divergence"""

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = True, softmax_gt: bool = True):
        super(BetaSkew, self).__init__(nclass, param, softmax_logits, softmax_gt)
        if len(self.param) <= 1 or not 0 < self.param[1] < 1:
            raise ValueError("Invalid beta parameter for Beta-Skew Divergence")
        self.beta = self.param[1]
        # Create KL instance only once during initialization
        self.kl = KL(nclass, param, softmax_logits=False, softmax_gt=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Use direct tensor operation for mixing
        mixed = logits * (1 - self.beta) + targets * self.beta
        return self.kl(mixed, targets)


class CauchySchwarz(BaseDivergence):
    """Cauchy-Schwarz Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)

        # Compute dot product of logits and targets - batchwise
        inner_prod = (logits * targets).sum(dim=1)

        # Compute squared L2 norms - batchwise
        logits_norm_sq = (logits * logits).sum(dim=1)
        targets_norm_sq = (targets * targets).sum(dim=1)

        # Compute the loss with clamping for numerical stability
        norm_product = torch.clamp(logits_norm_sq * targets_norm_sq, min=EPS_MEDIUM)
        cs_ratio = torch.clamp(inner_prod * inner_prod / norm_product, min=EPS_TINY)
        loss = -0.5 * torch.log(cs_ratio)
        return loss.mean()


# Dictionary-based factory defined at module level for efficiency
_LOSS_CLASSES = {
    "KL": KL,
    "TV": TV,
    "X2": X2,
    "PowD": PowD,
    "JS": JS,
    "GenKL": GenKL,
    "Exp": Exp,
    "LeCam": LeCam,
    "AlphaRenyi": AlphaRenyi,
    "BetaSkew": BetaSkew,
    "CauchySchwarz": CauchySchwarz
}


def get_loss(name: str, param: Optional[List[float]] = None, nclass: int = 10,
             softmax_logits: bool = True, softmax_gt: bool = True) -> BaseDivergence:
    """
    Factory function to create loss objects.

    Args:
        name: Name of the divergence measure
        param: Parameters for the divergence measure
        nclass: Number of classes
        softmax_logits: Whether to apply softmax to model predictions
        softmax_gt: Whether to apply softmax to ground truth distributions

    Returns:
        Instance of the specified divergence measure

    Raises:
        ValueError: If an invalid loss function name is provided
    """
    if name in _LOSS_CLASSES:
        return _LOSS_CLASSES[name](nclass, param, softmax_logits, softmax_gt)

    raise ValueError(f"Invalid loss function: {name}. Available options: {', '.join(_LOSS_CLASSES.keys())}")