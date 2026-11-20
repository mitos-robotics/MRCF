# Security policy

MRCF moves real robots, so this policy covers two kinds of problem: an
ordinary security hole, and a bug that can make the robot move when it should
not. Report both privately, never as a public issue, until a fix is out.

## What to report privately

**Security holes**, for example:

- anything that lets someone else on the network drive the robot or change
  what it is told to do, such as a port opened without being asked for, or a
  model server other than the one the user chose (`MRCF_MODEL_SERVER`,
  `MRCF_MODEL_URL`, or Ollama on this machine when neither is set)
- MRCF loading a policy package, its manifest or model weights in a way that
  runs code the user did not ask for, such as a pickle, `torch.load` without
  `weights_only`, or a package from a source other than the one named
- secrets (tokens, keys, credentials) written to the console, a log or the
  flight recorder, or the model server's key (`MRCF_MODEL_API_KEY`) sent
  anywhere other than that server

Installed plugins and model formats built on pickle (a Stable-Baselines3
`.zip`) run code by design, so install only policy packages and plugins you
trust. That is not a hole. MRCF doing it without being asked is.

**The robot moving when it should not**, for example:

- the safety layer letting through a plan or a step that its workspace
  bounds, joint limits, force limits or sequence rules should have blocked
- the interlock failing open, so the robot moves after a trip, or before every
  source has cleared
- the motion guard not pausing the robot for movement it should have seen, or
  a fail-safe watcher that crashed or lost its camera not stopping it
- a stop (`hold()`, `MotionStopped`) not reaching the motors

If you are not sure whether it counts, report it privately anyway. A pick that
misses, a misread board, or a wrong plan that the safety layer kept inside its
bounds is an ordinary bug: open an issue.

## How to report

Use either:

- **GitHub**: the repository's **Security** tab, then **Report a
  vulnerability** (GitHub's private vulnerability reporting). Only the
  maintainers see it. Maintainers: switch it on in Settings → Code security →
  Private vulnerability reporting.
- **Email**: [rango@mitosrobotics.com](mailto:rango@mitosrobotics.com).

Include as much of this as you can:

- the MRCF version or commit (`pip show mrcf`, `git rev-parse HEAD`)
- the backend (`MRCF_ARM`) and the deployment, and whether it happened on a
  real robot, a vendor's simulator, or offline
- what happened, and the steps that make it happen again
- for a motion problem, the flight recorder's incident, if there is one. It
  is a folder, `runs/flight/<session>/incidents/<incident>/` (under
  `MRCF_HOME` if that is set), and `python -m mrcf.flight_recorder --list`
  lists them. It holds camera frames. A calibrated camera's frames keep only
  the arm's work area, and every face found is covered (`mrcf/privacy.py`).
  Still, look through them first and leave out anything private (a hand, a
  screen, a face the model missed). Send it with the private report, never to
  a public issue.

The framework is designed to keep no pictures of people. A calibrated
camera's frames keep only what lies toward the arm's work volume, every face
is covered (once you fetch the face model with
`python tools/fetch_face_model.py`), and a computer's own cameras are never
opened. A saved picture with a person in it is a privacy bug: report it
here.

## What happens next

The maintainers acknowledge a report within 7 days. MRCF is maintained by a
small team, so a fix can take longer. You will hear whether the problem could
be reproduced and, once the maintainers know, when a fix is likely. The fix ships in a
release whose [changelog](CHANGELOG.md) says what it fixed, and credits you if
you want. Please wait for that release before saying anything publicly.

## Supported versions

| Version | Fixed |
|---|---|
| `main` | yes |
| the latest release | yes |
| older releases | no (upgrade) |

## MRCF is not a safety system

The motion guard watches fixed cameras and pauses the robot a fraction of a
second after something moves near it. It cannot see what its cameras cannot.
It is a guard, not a certified safety system, and MRCF is not for
safety-critical use. Nothing in MRCF replaces the robot's own stop, its
controller's limits, and a person within reach of the power switch. A failure
of the guard is still worth reporting as above, but no fix makes it a safety
system.
