## What and why

<!-- What this changes, and why. Link the issue it fixes, if there is one. -->

## How it was tested

<!-- What you ran, and where: offline, on a vendor's simulator, or on a real robot (which one).
     For a change to motion, what the robot did. -->

## Checklist

<!-- Tick what applies and is done. If something applies but is not done, say why. -->

- [ ] One logical change
- [ ] Tests pass (`pytest`)
- [ ] Lint passes (`ruff check .`)
- [ ] Type check passes (`mypy`)
- [ ] A change to motion is verified on a real robot, or on a vendor's simulator of one (URSim for a UR), and described above
- [ ] A change to `mrcf/backends/so101/` has had its tests run in a LeRobot environment
- [ ] The structure is kept ("The rules that keep the structure" in CONTRIBUTING.md): code moved to where it belongs and no test in `tests/test_architecture.py` or `tests/test_privacy.py` loosened, or a rule changed on purpose and explained above
- [ ] If shared code changed while working on one arm: no other arm's recording in `tests/baselines/` changed, or the change is behind a switch, or it was meant for that arm too and is explained above ("A change for one arm" in CONTRIBUTING.md)
- [ ] If the public API changed: `README.md` / `DEPLOYMENT_SPEC.md` / `POLICIES.md` updated
- [ ] A quirk or a limit the code cannot fix yet has an entry in the `NOTES.md` of the framework, robot or task it is about
- [ ] A change users will notice has a line under "Unreleased" in `CHANGELOG.md`
