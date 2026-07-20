import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import VF
from .sampling import save_samples


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def mnist_loader(
    root: str = ".", batch_size: int = 256, num_workers: int = 2
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    ds = datasets.MNIST(
        root, train=True, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train(
    epochs: int = 60, lr: float = 2e-4, data_root: str = ".", dev: str | None = None
) -> tuple[VF, AveragedModel, torch.Tensor]:
    dev = dev or device()
    dl = mnist_loader(data_root)

    model = VF().to(dev)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    z0 = torch.randn(64, 784, device=dev)  # fixed eval noise: comparable frames

    for epoch in range(epochs):
        total, seen = 0.0, 0
        for x1, _ in dl:  # labels discarded
            x1 = x1.to(dev).view(-1, 784) * 2 - 1
            x0 = torch.randn_like(x1)  # independent coupling
            t = torch.sigmoid(torch.randn(x1.size(0), device=dev))  # logit-normal t
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1  # linear conditional path
            loss = (model(xt, t) - (x1 - x0)).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update_parameters(model)
            total += loss.item() * x1.size(0)
            seen += x1.size(0)
        print(f"epoch {epoch:3d}  loss {total / seen:.4f}")
        save_samples(ema.module, z0)  # same name, auto-reloads

    return model, ema, z0
