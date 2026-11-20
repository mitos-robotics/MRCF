"""
llm_controller.py — the toolbench deployment's controller (the UR5 putting tools into a box): the
framework's table controller (mrcf/tabletop/controller.py), run with this
deployment's config (config.py).

Usage (from the repository root):
    python run.py --deployment toolbench
    python -m deployments.toolbench.llm_controller
"""

import sys as _sys

if not __package__:
    _sys.exit("Run the toolbench controller from the repository root as a module: "
              "python -m deployments.toolbench.llm_controller (or python run.py).")

from . import skills  # noqa: E402,F401 — binds the table task to this deployment; its skills registered
from mrcf.tabletop import controller as _controller  # noqa: E402


def main() -> None:
    """The table controller, for this deployment."""
    _controller.main()


if __name__ == "__main__":
    main()
