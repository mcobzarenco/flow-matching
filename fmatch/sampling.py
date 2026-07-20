import torch
import torch.nn as nn
from torchvision import utils
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


@torch.no_grad()
def sample(
    net: nn.Module, x0: torch.Tensor, steps: int = 50
) -> torch.Tensor:  # Heun; integrate from x0
    net = net.eval()
    n, dt = x0.size(0), 1 / steps
    x = x0.clone()

    def v(x: torch.Tensor, t: float) -> torch.Tensor:
        return net(x, torch.full((n,), t, device=x0.device))

    for i in range(steps):
        v0 = v(x, i * dt)
        if i == steps - 1:  # final step: posterior-mean jump
            x = x + dt * v0
            break
        x = x + dt * (v0 + v(x + dt * v0, (i + 1) * dt)) / 2
    return ((x.view(n, 1, 28, 28) + 1) / 2).clamp(0, 1)


def save_samples(
    net: nn.Module,
    x0: torch.Tensor,
    path: str = "samples.png",
    steps: int = 50,
    scale: int = 4,
) -> None:
    img = utils.make_grid(sample(net, x0, steps), nrow=8)
    img = resize(
        img,
        [img.shape[-2] * scale, img.shape[-1] * scale],
        interpolation=InterpolationMode.NEAREST_EXACT,
    )
    utils.save_image(img, path)
