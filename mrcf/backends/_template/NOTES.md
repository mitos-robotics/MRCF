# My Arm — notes

What this robot has shown that its code and config cannot say: quirks, failure
modes, what to watch for, what was tried. One entry per issue; delete an
entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).

- This robot, on any rig and any task — here.
- One task, on any robot — `deployments/<task>/NOTES.md`.
- Only this rig: its table, its light, its room — `rig/<backend>/notes.md` (not in git).

Each entry:

```markdown
### What happens, in a few words
- **Seen:** the date; what was seen; the flight recorder's incident, if there is one
  (`runs/flight/<session>/incidents/<incident>/`)
- **Cause:** what is behind it, or "not known yet" and the best guess
- **Workaround:** what to do until it is fixed
- **Status:** open, or fixed (the commit, or where the fix lives)
```

## Limits

What this robot cannot do that others can, by design or not yet (no force
sensing, a reach smaller than the work surface), one or two sentences each.
The framework's own limits are in `mrcf/NOTES.md`.

(nothing yet)

## Open

(nothing yet)
