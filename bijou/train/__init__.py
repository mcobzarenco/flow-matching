"""Train a Bijou VLA family on a LeRobot v3 dataset.

Run with ``python -m bijou.train`` -- see ``bijou/train/__main__.py``.
A run trains ONE model family (``--family``, §5 of
docs/architecture.md). The package splits by concern: the flag surface
and its resolution in ``args.py``, the step/probe machinery in
``loop.py``, checkpoint capture/write in ``saving.py``, and ``main()``
with the orchestration that ties them together in ``cli.py``. The
training API is re-exported here.
"""

from .args import (
    ARCH_DEFAULTS,
    ARCH_FLAGS,
    DEFAULT_BACKBONE,
    ArchSection,
    CheckpointResolution,
    TrainArgs,
    _build_parser,
    parse_args,
    reconcile_lr_offer,
)
from .cli import (
    apply_adamc_weight_decay,
    backbone_group_index,
    build_optimizer_param_groups,
    check_resume_seed,
    decay_split,
    load_family_weights,
    lr_lambda,
    main,
    rehome_fused_step_tensors,
    resume_hyperparameter_notes,
    stage2_backbone_init,
)
from .loop import (
    ChunkedBatch,
    ChunkingCollator,
    allreduce_gradients,
    broadcast_module_states,
    summed_loss_counts,
)
from .saving import (
    Normalizer,
    Normalizers,
    TrainState,
    build_vla_metadata,
    capture_checkpoint_tensors,
    save_checkpoint,
    write_checkpoint,
)

__all__ = [
    "ARCH_DEFAULTS",
    "ARCH_FLAGS",
    "DEFAULT_BACKBONE",
    "ArchSection",
    "CheckpointResolution",
    "ChunkedBatch",
    "ChunkingCollator",
    "Normalizer",
    "Normalizers",
    "TrainArgs",
    "TrainState",
    "_build_parser",
    "allreduce_gradients",
    "apply_adamc_weight_decay",
    "backbone_group_index",
    "broadcast_module_states",
    "build_optimizer_param_groups",
    "build_vla_metadata",
    "capture_checkpoint_tensors",
    "check_resume_seed",
    "decay_split",
    "load_family_weights",
    "lr_lambda",
    "main",
    "parse_args",
    "reconcile_lr_offer",
    "rehome_fused_step_tensors",
    "resume_hyperparameter_notes",
    "save_checkpoint",
    "stage2_backbone_init",
    "summed_loss_counts",
    "write_checkpoint",
]
