"""
mrcf — the Modular Robotics Control Framework: control a robot in plain English
with local models, planned and learned policies behind a safety layer.

The one name `pip install` puts in site-packages. Everything is a module or
package under it:

    from mrcf import manipulation, safety
    from mrcf.policies.learned import LearnedPolicy
    from mrcf import backends; backends.arm()        # the robot that runs (MRCF_ARM)

The scripts beside it in the repository — run.py, tools/, deployments/ — use
it; they are not installed. A rig's own files are in rig/<backend>/
(rig_state.py), never in the package.
"""

__version__ = "0.1.0"
