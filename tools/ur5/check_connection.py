"""
check_connection.py — can this computer reach the UR5, and is the arm in a
state to be driven? MOVES NOTHING.

    python tools/ur5/check_connection.py                  # the controller at robot_config.HOST (rig/ur5/host)
    python tools/ur5/check_connection.py --host 192.168.x.y   # another controller
    python tools/ur5/check_connection.py --pose home      # ... and the joints now as HOME_JOINTS_DEG

In order, each saying what to fix when it fails:

  1. The network: which of this computer's addresses the controller is reached
     from, and its ports — the dashboard (29999), the script port (30002),
     RTDE (30004) and the Robotiq URCap (63352, when there is a gripper).
  2. The dashboard: PolyScope's version, the model, the robot and safety modes,
     the program loaded on the pendant.
  3. RTDE, reading only: the joints, the flange against this repository's
     kinematics (kinematics.py), the speed slider, the force at the tool.
  4. The Robotiq, reading only: activated, its position, whether it holds something.
  5. --pose NAME: the joints now, as the line to paste into robot_config
     (jog the arm there on the pendant first).

Reading only: the dashboard's queries and RTDEReceiveInterface. It never opens
RTDEControlInterface — that uploads a script, which stops any program running
on the pendant — and never activates the gripper, which moves its fingers.
MRCF_UR5_HOST=offline runs it on offline.py's UR5.
"""

import argparse
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from tools.ur5._common import rc  # noqa: E402

PORTS = {29999: "the dashboard", 30002: "the script port (ur_rtde uploads its control script here)",
         30004: "RTDE (every reading and command)"}
ROBOTIQ_PORT = 63352
SUPPLY_LOW_V = 46.5          # between this arm's healthy 47.2-47.4 V and its failing 45.0-45.8 V


def say(ok: "bool | None", text: str) -> None:
    print(f"  {'ok ' if ok else '-- ' if ok is None else 'NO '} {text}")


def check_network(host: str) -> bool:
    print(f"\n1. The network, to {host}")
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect((host, 9))                     # no packet is sent: this only asks for the route
        local = probe.getsockname()[0]
        probe.close()
    except OSError as exc:
        say(False, f"no route to {host}: {exc}")
        local = None
    if local is not None:
        same = local.rsplit(".", 1)[0] == host.rsplit(".", 1)[0]
        say(same, f"this computer reaches it from {local}" + ("" if same else
            f" — not on {host.rsplit('.', 1)[0]}.x: give the Ethernet adapter a fixed address there "
            f"(System Settings > Network > the adapter > Details > TCP/IP > Manually, "
            f"{host.rsplit('.', 1)[0]}.100, mask 255.255.255.0). If Wi-Fi is on the same range, use another "
            "range for the robot."))
    ok = True
    ports = dict(PORTS)
    if rc.GRIPPER == "robotiq":
        ports[ROBOTIQ_PORT] = "the Robotiq URCap's socket"
    for port, what in ports.items():
        try:
            socket.create_connection((host, port), timeout=2.0).close()
            say(True, f"{port}: {what}")
        except OSError as exc:
            ok = ok and port == ROBOTIQ_PORT
            hint = ("is the URCap installed and the gripper on?" if port == ROBOTIQ_PORT else
                    "is the cable in, the controller booted, and its address the one above "
                    "(pendant: Setup Robot > Network)?")
            say(False, f"{port}: {what} — {exc.__class__.__name__}: {hint}")
    return ok


def check_dashboard(dash) -> None:
    print("\n2. The controller (dashboard)")
    for label, ask in (("PolyScope", "polyscopeVersion"), ("model", "getRobotModel"),
                       ("serial", "getSerialNumber"), ("robot mode", "robotmode"),
                       ("safety mode", "safetymode"), ("program", "getLoadedProgram"),
                       ("program state", "programState")):
        try:
            say(None, f"{label}: {str(getattr(dash, ask)()).strip()}")
        except Exception as exc:                    # noqa: BLE001 — some queries are newer than 3.15
            say(None, f"{label}: (not answered: {exc})")


def check_state(receive) -> "np.ndarray":
    from mrcf.backends.ur5 import kinematics
    from mrcf.backends.ur5.arm import JOINT_MODES, ROBOT_MODES, SAFETY_MODES, pose_to_matrix
    from mrcf.backends.ur5.env import JOINT_NAMES

    print("\n3. The arm (RTDE, reading only)")
    mode, safety = int(receive.getRobotMode()), int(receive.getSafetyMode())
    say(mode == 7, f"robot mode: {ROBOT_MODES.get(mode, mode)}"
        + ("" if mode == 7 else " — on the pendant: the initialization screen, ON, then START"))
    if hasattr(receive, "getActualMainVoltage"):
        # The control box's 48 V supply. This arm read 47.2-47.4 V while it worked and 45.0-45.8 V
        # when its joints stopped finishing their boot (2026-11-05, mrcf/backends/ur5/NOTES.md).
        supply = float(receive.getActualMainVoltage())
        if supply > 1.0:
            say(supply >= SUPPLY_LOW_V, f"48 V supply: {supply:.1f} V" + ("" if supply >= SUPPLY_LOW_V else
                f" — below {SUPPLY_LOW_V:.1f} V: a weak power supply in the control box can leave the joints "
                "stuck booting; have it measured"))
    if mode in (4, 5) and hasattr(receive, "getJointMode"):          # powering on, or idle: what each joint says
        joints = [JOINT_MODES.get(int(m), str(int(m))) for m in receive.getJointMode()]
        volts = float(receive.getActualRobotVoltage()) if hasattr(receive, "getActualRobotVoltage") else None
        print("     joints say: " + ", ".join(f"{n} {j}" for n, j in zip(JOINT_NAMES, joints))
              + ("" if volts is None else f"; the arm gets {volts:.0f} V"))
        if mode == 4 and set(joints) == {"booting"}:
            joint_v = np.asarray(receive.getActualJointVoltage(), float) if hasattr(receive, "getActualJointVoltage") \
                else np.zeros(6)
            if not joint_v.any():                                   # nothing back from any joint
                say(False, "no joint has answered the controller"
                    + (" though the arm has its power" if volts and volts > 40 else "")
                    + ": if it stays so for a minute, OFF, shut the controller down, and reseat the arm cable "
                      "(controller off); the pendant's Log tab names what it waits for")
            else:
                temps = np.asarray(receive.getJointTemperatures(), float) if hasattr(receive, "getJointTemperatures") \
                    else None
                say(False, f"the joints answer ({joint_v.min():.0f}-{joint_v.max():.0f} V"
                    + ("" if temps is None else f", {temps.min():.0f}-{temps.max():.0f} °C")
                    + ") but have not finished booting: the pendant's Log tab says what they wait for")
    say(safety in (1, 2), "safety: " + ({1: "normal", 2: "reduced"}.get(safety) or SAFETY_MODES.get(safety, str(safety))))
    q = np.asarray(receive.getActualQ(), float)
    print("     joints (deg): " + ", ".join(f"{n} {v:7.2f}" for n, v in zip(JOINT_NAMES, np.degrees(q))))
    tcp = pose_to_matrix(receive.getActualTCPPose())
    print(f"     TCP (m): {np.round(tcp[:3, 3], 4).tolist()}, pointing {np.round(tcp[:3, 2], 3).tolist()}")
    flange = kinematics.fk(q)
    off = flange[:3, :3].T @ (tcp[:3, 3] - flange[:3, 3])          # the TCP in the flange frame
    if np.linalg.norm(off) < 0.003:
        say(True, f"kinematics.py agrees with the controller: {np.linalg.norm(off) * 1000:.1f} mm apart")
    else:
        say(None, f"the controller's TCP is {np.round(off * 1000, 1).tolist()} mm from the flange "
                  "(flange frame): the pendant has a TCP set. Fine — the backend sets its own (the flange) "
                  "as it connects. If that is not the tool's offset, the kinematics disagree.")
    scaling = float(receive.getTargetSpeedFraction())
    say(scaling <= 0.3, f"speed slider: {scaling * 100:.0f}%" + ("" if scaling <= 0.3 else
        " — for the first moves, set it to 10-25% on the pendant"))
    force = np.asarray(receive.getActualTCPForce(), float)
    say(None, f"force at the tool: {np.linalg.norm(force[:3]):.1f} N (a CB3's is estimated, so a few N at rest is normal)")
    return q


def check_gripper(host: str) -> None:
    from mrcf.backends.ur5.gripper import RobotiqSocket

    print("\n4. The gripper (reading only)")
    if rc.GRIPPER is None:
        say(None, "none configured (robot_config.GRIPPER is None)")
        return
    grip = RobotiqSocket(host, max_open_m=rc.GRIPPER_MAX_OPEN_M, pos_open=rc.ROBOTIQ_POS_OPEN,
                         pos_closed=rc.ROBOTIQ_POS_CLOSED)
    try:
        grip._sock = socket.create_connection((host, RobotiqSocket.PORT), timeout=2.0)   # without connect()'s checks
        act, sta, pos, obj = (grip.get(v) for v in ("ACT", "STA", "POS", "OBJ"))
    except Exception as exc:                        # noqa: BLE001
        say(False, f"no answer from the Robotiq URCap: {exc}")
        return
    finally:
        grip.disconnect()
    say(act == 1 and sta == 3, "activated" if (act == 1 and sta == 3) else
        f"not activated (ACT {act}, STA {sta}): activate it on the pendant (Installation > Gripper)")
    say(None, f"position {pos} (0-255) = {grip._width_for(pos) * 1000:.0f} mm by ROBOTIQ_POS_OPEN "
              f"{rc.ROBOTIQ_POS_OPEN} / _CLOSED {rc.ROBOTIQ_POS_CLOSED}; holding: {obj in (1, 2)}")
    print("     To measure those two: open it fully from the pendant and run this again (POS is _OPEN), "
          "then close it fully with nothing between the fingers (POS is _CLOSED).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--host", default=rc.HOST, help=f"the controller (robot_config.HOST: {rc.HOST})")
    parser.add_argument("--pose", metavar="NAME", help="print the joints now as NAME_JOINTS_DEG (home, park)")
    args = parser.parse_args()

    offline = args.host is None
    print(f"UR5 at {'offline (in memory)' if offline else args.host} — reading only, nothing moves.")
    if offline and rc.HOST_SOURCE is None:
        print(f"  ({rc.NO_HOST_HINT})")
    if offline:
        from mrcf.backends.ur5.offline import OfflineController, OfflineDashboard
        dash, receive = OfflineDashboard(), OfflineController()
    else:
        if not check_network(args.host):
            sys.exit("\nThe controller is not reachable: fix the network first (above).")
        try:
            import dashboard_client
            import rtde_receive
        except ImportError:
            sys.exit("ur_rtde is not installed in this Python: see mrcf/backends/ur5/NOTES.md.")
        dash = dashboard_client.DashboardClient(args.host)
        dash.connect()
        receive = rtde_receive.RTDEReceiveInterface(args.host)
    try:
        check_dashboard(dash)
        q = check_state(receive)
        if not offline:
            check_gripper(args.host)
    finally:
        receive.disconnect()
        dash.disconnect()
    if args.pose:
        print(f"\n5. Paste into mrcf/backends/ur5/robot_config.py:\n\n"
              f"{args.pose.upper()}_JOINTS_DEG = ({', '.join(f'{v:.2f}' for v in np.degrees(q))})\n")


if __name__ == "__main__":
    main()
