"""``python -m bijou.eval`` entry point.

Definition-free on purpose: spawned dataloader workers cannot unpickle
objects defined in a package's ``__main__`` module (CPython refuses to
re-import ``pkg.__main__`` in children because its code runs on import),
so everything lives in ``bijou.eval.cli``.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
