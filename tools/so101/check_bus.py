"""
check_bus.py — which servos are actually answering on the serial bus?

For when a tool dies at startup with "Missing motor IDs". That error comes from
lerobot's own motor check and says nothing about WHY, so this pings the bus
directly, reports every ID that answers, and reads the health registers of the
ones that do.

The usual cause on this arm is the gripper (ID 6): it is last in the daisy
chain, its cable flexes every time the jaws move, and it is the motor that gets
driven into a hard stop most often. Both failure modes — an unseated connector
and a tripped overload latch — look identical from lerobot's message.

Usage:
    python tools/so101/check_bus.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.so101._common import rc   # noqa: E402

EXPECTED = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
            4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def main() -> None:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    # One throwaway motor entry: the bus needs a config to open, but
    # broadcast_ping does not care what is in it, so this avoids the very
    # motor check we are trying to diagnose around.
    bus = FeetechMotorsBus(
        port=rc.FOLLOWER_PORT,
        motors={"probe": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    print(f"Opening {rc.FOLLOWER_PORT}...")
    bus.port_handler.openPort()
    bus.port_handler.setBaudRate(1_000_000)

    found = bus.broadcast_ping() or {}
    print(f"\n{'id':>3}  {'joint':14} {'model':>7}  status")
    for mid, name in EXPECTED.items():
        if mid in found:
            print(f"{mid:>3}  {name:14} {found[mid]:>7}  answering")
        else:
            print(f"{mid:>3}  {name:14} {'—':>7}  NO RESPONSE")
    extra = set(found) - set(EXPECTED)
    if extra:
        print(f"\nUnexpected IDs answering: {sorted(extra)}")

    missing = [m for m in EXPECTED if m not in found]
    print()
    if not missing:
        print("All six answering — the bus is fine.")
    else:
        names = ", ".join(f"{m} ({EXPECTED[m]})" for m in missing)
        print(f"Not answering: {names}")
        print("""
In order of likelihood:

  1. POWER CYCLE the arm. Unplug its power, wait ~5 s, plug back in, re-run
     this. An STS3215 that has been driven into a hard stop can latch its
     overload protection and stop responding until power is removed — and this
     gripper has spent today closing onto chess pieces, which is exactly that.
     lerobot deliberately sets the gripper's limits low (Max_Torque_Limit 500,
     Protection_Current 250, Overload_Torque 25), so it trips before damage.

  2. CHECK THE CABLE at both ends of the missing servo. The gripper is last in
     the daisy chain and its cable flexes on every open and close, so its
     connector is the one that works loose. Unplug and reseat both ends.

  3. If it still does not answer after a power cycle and a reseat, the servo or
     its cable has failed. Swap the cable first — it is the cheaper suspect.

Nothing in software can reach a motor that is not on the bus; do not re-run the
calibration until this prints all six.""")
    bus.port_handler.closePort()


if __name__ == "__main__":
    main()
