# The table task — notes

What the table task (every deployment that does it) has shown that its code and config cannot say. One entry per
issue; delete an entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).

Each entry: **Seen** (the date, what, the flight recorder's incident),
**Cause**, **Workaround**, **Status** (open, or fixed and where).

Each deployment that does it has its own notes too: deployments/tabletop (the
UR5's letter blocks), deployments/tidy (the SO-101 putting things away),
deployments/toolbench (the UR5 putting hand tools into a box).

## Limits

- **Upright objects first; boxes squared.** The pick grips across an object's
  foot with the jaws turned as the arm turns them — except a known box kind
  (KNOWN_SIZES_MM: blocks), whose jaws turn to close across its sides, and,
  where a deployment's rules say so, a long thing (CLOSE_ACROSS_LONG: closed
  across it; LYING_PICK: a tool lying along the table, gripped across by its
  handle or at its balance point). Otherwise a pen lying at a slant is gripped
  as it happens to lie, if at all.
- **Names come from one look.** Each object is named once, from one crop, and
  again only when a scan matched it by its look alone (moved by hand, back
  after it had gone). A name moves to another remembered object only when the
  vision model reads that one's name on it *and* the camera's colour agrees.
  Letter blocks carry a different letter on each face, and the model misreads
  them (2026-11-12: red U read as D, and as P). So a block turned over by hand
  may keep an old letter in its name. A wrong name stays until the user says
  otherwise (identify), or `forget`.
- **Left and right are one viewer's.** LEFT_RIGHT_SEEN_FROM: a person across
  the table, facing the arm, unless set to the arm.
- **A container is outlined by one look** of the overhead camera (place_in,
  where a deployment's arm can put things into containers: GripperSpec.place_in);
  one it cannot see whole (out of its view, under the arm) is outlined short.

## Open

- **A place on a taken spot goes home and tries again, not beside it.**
  - **Seen:** 2026-11-02, offline, gemma4:31b. "Put the green box on the
    shelf" was asked with the red bottle standing on the shelf spot. The place
    was refused (no room for the jaws). The planner sent the arm home and asked
    again, was refused again, then planned the same thing a third time, and
    the controller handed back to the user.
  - **Cause:** the controller allows a refused step one more go after a home,
    for an arm whose reach depends on where it starts. The planner takes that
    go even when the reason was "taken", and does not think of `place_beside`.
  - **Workaround:** say where instead: "put it next to the bottle".
  - **Status:** open. It costs one wasted home.
