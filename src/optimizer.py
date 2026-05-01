"""Optimizer construction for AdamW and Muon/AdamW hybrid training.

References:
    - The Muon split between hidden-layer matrices and everything else follows
      the modded-nanogpt training recipes published by Keller Jordan and
      collaborators.
    - The Newton-Schulz orthogonalization update is adapted from the public
      Muon writeup and reference implementation lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.utils import OptimizerConfig


@torch.no_grad()
def zeroth_power_via_newton_schulz5(
    grad_matrix: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the zeroth matrix power used by Muon.

    This iteration mirrors the public Muon reference: normalize the gradient,
    run a small fixed number of Newton-Schulz steps, and project back to the
    original shape.
    """

    if grad_matrix.ndim != 2:
        raise ValueError("Muon expects a rank-2 gradient matrix.")
    a, b, c = (3.4445, -4.7750, 2.0315)
    working_dtype = torch.bfloat16 if grad_matrix.is_cuda else torch.float32
    x = grad_matrix.to(dtype=working_dtype) / (grad_matrix.norm() + eps)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.transpose(0, 1)
        transposed = True
    for _ in range(steps):
        gram = x @ x.transpose(0, 1)
        correction = b * gram + c * gram @ gram
        x = a * x + correction @ x
    if transposed:
        x = x.transpose(0, 1)
    return x.to(dtype=grad_matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-valued hidden-layer weights."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float,
        nesterov: bool,
        weight_decay: float,
        ns_steps: int,
        eps: float,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Perform a Muon optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(grad)
                    state["momentum_buffer"] = momentum_buffer
                momentum_buffer.mul_(momentum).add_(grad)
                update = grad.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
                update_matrix = update if update.ndim == 2 else update.reshape(update.shape[0], -1)
                projected = zeroth_power_via_newton_schulz5(
                    update_matrix,
                    steps=ns_steps,
                    eps=eps,
                ).reshape_as(update)
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(projected, alpha=-lr)
        return loss


@dataclass(slots=True)
class HybridOptimizer:
    """Thin wrapper exposing a single optimizer-like interface."""

    muon: Muon
    adamw: torch.optim.AdamW

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients across both underlying optimizers."""

        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Step both underlying optimizers."""

        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state."""

        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state."""

        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])

    def set_learning_rates(self, *, adamw_lr: float, muon_lr: float) -> None:
        """Update both optimizer learning rates."""

        for group in self.adamw.param_groups:
            group["lr"] = adamw_lr
        for group in self.muon.param_groups:
            group["lr"] = muon_lr


def _split_hybrid_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into Muon, AdamW-decay, and AdamW-no-decay groups."""

    muon_params: list[torch.nn.Parameter] = []
    adamw_decay_params: list[torch.nn.Parameter] = []
    adamw_no_decay_params: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        is_embedding_or_head = (
            "token_embedding" in lowered or "lm_head" in lowered or "embed" in lowered
        )
        if parameter.ndim >= 2 and not is_embedding_or_head:
            muon_params.append(parameter)
        elif parameter.ndim >= 2:
            adamw_decay_params.append(parameter)
        else:
            adamw_no_decay_params.append(parameter)

    return muon_params, adamw_decay_params, adamw_no_decay_params


def _split_adamw_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into AdamW decay and no-decay groups."""

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)
    return decay_params, no_decay_params


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer | HybridOptimizer:
    """Build the configured optimizer."""

    optimizer_name = config.name.lower()
    if optimizer_name == "adamw":
        decay_params, no_decay_params = _split_adamw_parameters(model)
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.adamw_lr,
            betas=config.betas,
            eps=config.eps,
        )

    if optimizer_name != "muon_hybrid":
        raise ValueError(f"Unsupported optimizer: {config.name}")

    muon_params, adamw_decay_params, adamw_no_decay_params = _split_hybrid_parameters(model)
    muon = Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=config.momentum,
        nesterov=config.nesterov,
        weight_decay=config.weight_decay,
        ns_steps=config.muon_ns_steps,
        eps=config.eps,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": adamw_decay_params, "weight_decay": config.weight_decay},
            {"params": adamw_no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.adamw_lr,
        betas=config.betas,
        eps=config.eps,
    )
    return HybridOptimizer(muon=muon, adamw=adamw)
