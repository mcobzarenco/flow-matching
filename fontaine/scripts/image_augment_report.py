"""Browsable HTML report of the --image-augment v0 recipe (owner ask
2026-08-17 07:55:24Z): real frames sampled from a demo dataset, each
shown next to several augmented draws, so the recipe can be judged by
eye against the frozen pre-reg
(posts/2026-08-15-prereg-image-augment-sim2real.md).

Frames come straight off the dataset's encoded videos (the same pixels
training decodes); draws go through bijou.modelling.image_augment
verbatim — the collator's exact call, seeded, so the grid is
reproducible.

Usage (wherever the dataset lives):
  uv run python fontaine/scripts/image_augment_report.py \
      --root ~/datasets/fontaine/grasp_demos_v2/merged \
      --out reports/augment__image_augment_v0_grid.html \
      [--per-camera 3] [--draws 4] [--seed 0]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CAMERAS = ("observation.images.front", "observation.images.wrist")

HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>--image-augment v0: sampled frames vs augmented draws</title>
<style>
body { background:#111418; color:#d8dee6; font:14px/1.5 system-ui, sans-serif;
       margin:2rem auto; max-width:1400px; padding:0 1rem; }
h1 { font-size:1.3rem; } h2 { font-size:1.05rem; margin-top:2rem; color:#9fb2c8; }
p.meta { color:#8494a7; }
table { border-collapse:collapse; margin:0.5rem 0 1.5rem; }
td { padding:4px; vertical-align:top; text-align:center; }
td img { width:320px; height:240px; display:block; border-radius:4px; }
td.label { color:#8494a7; font-size:12px; padding-top:2px; }
.orig img { outline:2px solid #4f8cc9; }
code { color:#a8c7e8; }
</style></head><body>
"""


def frame_to_jpeg_b64(array, quality: int = 82) -> str:  # noqa: ANN001
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def decode_frame(path: Path, index: int):  # noqa: ANN201 — np.ndarray, np deferred
    import av

    container = av.open(str(path))
    for i, frame in enumerate(container.decode(video=0)):
        if i == index:
            container.close()
            return frame.to_ndarray(format="rgb24")
    container.close()
    raise SystemExit(f"{path}: frame {index} past end")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-camera", type=int, default=3)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    from bijou.modelling.image_augment import DEFAULT_SPEC, augment_image

    root = args.root.expanduser()
    provenance = json.loads(
        (root / "meta" / "demo_provenance.json").read_text(),
    )

    parts: list[str] = [HTML_HEAD]
    parts.append("<h1>--image-augment v0: dataset frames vs augmented draws</h1>")
    parts.append(
        f"<p class='meta'>dataset <code>{root.name}</code> "
        f"(kept {provenance.get('kept')}, expert <code>"
        f"{provenance.get('expert_head')}</code>) — recipe "
        "<code>bijou.modelling.image_augment</code> v0 (pre-reg "
        "2026-08-15), collator-identical call, seed "
        f"{args.seed}. Blue outline = the original frame; each row's "
        f"{args.draws} draws re-run the full 14-draw parameter block "
        "(crop/translate → photometric → defocus → noise → JPEG).</p>",
    )

    for camera in CAMERAS:
        parts.append(f"<h2>{camera}</h2>")
        video_dir = root / "videos" / camera / "chunk-000"
        files = sorted(video_dir.glob("file-*.mp4"))
        if not files:
            raise SystemExit(f"no videos under {video_dir}")
        rng = torch.Generator().manual_seed(args.seed)
        for k in range(args.per_camera):
            pick = files[int(torch.randint(len(files), (1,), generator=rng).item())]
            index = int(torch.randint(200, (1,), generator=rng).item()) + 30
            frame = decode_frame(pick, index)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            cells = [
                (
                    f"<td class='orig'><img src='data:image/jpeg;base64,"
                    f"{frame_to_jpeg_b64(frame)}'>"
                    f"<div class='label'>{pick.name} frame {index}</div></td>"
                ),
            ]
            for d in range(args.draws):
                gen = torch.Generator().manual_seed(
                    args.seed * 10_000 + k * 100 + d,
                )
                out = augment_image(tensor, gen, DEFAULT_SPEC)
                array = (out.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                cells.append(
                    f"<td><img src='data:image/jpeg;base64,"
                    f"{frame_to_jpeg_b64(array)}'>"
                    f"<div class='label'>draw {d}</div></td>",
                )
            parts.append("<table><tr>" + "".join(cells) + "</tr></table>")

    parts.append("</body></html>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(parts))
    print(f"[report] {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
