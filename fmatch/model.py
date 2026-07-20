import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):  # pre-LN residual GEGLU block
    def __init__(self, h: int, mult: int = 2) -> None:
        super().__init__()
        g = max(int(2 * mult * h / 3), 1)  # inner width, 2/3 convention
        self.norm = nn.LayerNorm(h)
        self.gate_val = nn.Linear(h, 2 * g)  # fused gate+value projection
        self.proj = nn.Linear(g, h)
        nn.init.zeros_(self.proj.weight)  # block is identity at init
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.gate_val(self.norm(x)).chunk(2, dim=-1)
        return x + self.proj(F.gelu(a, approximate="tanh") * b)


class VF(nn.Module):  # v_theta(x, t) — unconditional
    freqs: torch.Tensor  # buffer; declared for the type checker

    def __init__(
        self, d: int = 784, h: int = 1024, tdim: int = 64, depth: int = 6
    ) -> None:
        super().__init__()
        self.register_buffer(
            "freqs", torch.randn(tdim // 2) * 10
        )  # Fourier time embedding
        self.inp = nn.Linear(d + tdim, h)
        self.blocks = nn.Sequential(*[Block(h) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = 2 * torch.pi * t[:, None] * self.freqs[None]
        return self.out(self.blocks(self.inp(torch.cat([x, a.sin(), a.cos()], -1))))
