# UR5 — notes

What this robot has shown that its code and config cannot say: quirks, failure
modes, what to watch for, what was tried. One entry per issue; delete an
entry once it is fixed (git keeps it). What belongs where:
[DEPLOYMENT_SPEC.md](../../../DEPLOYMENT_SPEC.md#what-you-learn-and-where-it-goes).
Measured values and the reasons behind them stay in `robot_config.py`, beside
the values.

Each entry: **Seen** (the date, what, the flight recorder's incident),
**Cause**, **Workaround**, **Status** (open, or fixed and where).
The framework's own limits, on every robot, are in [mrcf/NOTES.md](../../NOTES.md).

## Limits

- **Not yet done from the 2026-11-12 safety review** (each needs the arm to try):
  no RTDE watchdog (a hung Mac or a pulled cable leaves a move to finish by
  itself); a learned policy's stream has no per-tick speed cap (servoL); the
  gripper box does not lengthen for what is held, and go_home/go_park rise
  only to TRANSIT_HEIGHT_M whatever stands on the table; wrist 3 has only the
  ±360° joint limit, so turns can wind the Hand-E's cable.

- **Three days on the arm.** 2026-11-12: `tools/ur5/check_motion.py` (lines, a
  stop, a pause, freedrive, home and park), the overhead camera calibrated by
  hand, the sentry with a hand in view, then letter blocks picked, placed and
  stacked three high by typed command (two commands for the tower);
  2026-11-13: the blocks again, a tower in one command; 2026-11-20: hand tools
  into a box (deployments/toolbench). The sweeps' changes of 2026-11-12 (the
  force stop on every move, the stop on Ctrl-C, the jaws halted by a pause,
  the base clearance, the check after a place) were in place for the later
  runs.
- **It grips nothing wider than about 48 mm.** The Hand-E's fingertips open
  52 mm (measured), less the slack a pick leaves either side: a bottle or a
  box is gripped across its narrow side or not at all; a block across its
  faces, not its corners (a 40 mm cube is 57 mm across them).
- **No wrist camera.** Picks line up with the overhead camera alone; the wrist
  camera's calibrations are still the SO-101's tools.
- **Its force reading is rough.** A CB3 estimates the force at the tool from
  its joint currents. Its change since the move began (`contact_wrench`)
  swung 35-74 N in free air in check_motion's moves, so the safety monitor
  stops a skill at 100 N (`CONTACT_FORCE_LIMITS`): it catches a collision, not
  a press. On 2026-11-12 it stopped a descent onto a dropped block at 100.9 N,
  after the person at the arm had reached for the e-stop.
- **Not every path is a straight line.** `move_ee_to` is a straight line
  (moveL) with the orientation held, except a move across at full speed (5 cm
  or more sideways, both ends 5 cm or more up), which is one joint move where
  its path keeps the jaws pointing down (`TRANSIT_*`). Turning the jaws (more
  than 5°), `go_home` and `go_park` are joint moves (moveJ), whose path between its ends
  is not a line. A turn happens where the grasp point already is, and home and
  park rise to the transit height first.

## Open

### ON stops at "Power on": the joints never finish booting
- **Seen:** 2026-11-05 and -09, every power-on. The initialization
  screen stays at "Power on" with START greyed; RTDE says robot mode 4 and all
  six joints in joint mode 247 (booting); URControl logs `POWER OFF -> POWER
  ON` and nothing after. A good start on this arm goes `POWER ON -> IDLE` in
  ~2.7 s, then `hasJointsSentData`. It last reached Idle and Running on
  2026-11-02 18:35 (controller clock, in /root/log_history.txt).
- **Seen in Low Level Control** (2026-11-09; Expert Mode, UR service manual
  3.1.2 sections 3.1.16 and 5.3.4): "Turn power on" brings all six joints up in
  BOOTLOADER with 45 V on the arm; "Go to Idle" says "Starting main program
  for all joints...", all six reach BOOTING and each reports its torque
  constant (K_tau about 0.128 for the three big joints, 0.094 for the
  wrists: the right sizes), then "FAILURE: The command has timed out!". With
  it, C4A91 (packet counter, joint 5) and C4A77 ("lost package from uPA to
  joints": uPA is the safety control board's processor). Every joint answers
  and every one stalls at the same step: not one joint. Low Level Control
  (URLLControl) writes no log file; the pendant's screen is the only record.
- **Cause:** a fault in the control box, not yet found: most likely the
  safety control board (it drives the joints' shared line), else the 48 V
  supply or the current distributor (the UR service manual's three suspects).
  Against a joint or the arm: all six fail alike. Two more signs, neither
  proof: the safety board starts in `scb_mode AB_FAULT` on 2026-11-05 and
  -09 and never in the two kept 2026-09-29 reports (it may be left over from
  the stuck session before); and the supply URControl logs with the arm
  unpowered (`EnergyEaterMonitor::zeroCalibration ... supply voltage was`)
  read 47.21-47.36 V on 2026-09-17/29 and 44.4-45.8 V since 2026-11-05.
- **Not the cause:** the controller's software. On 2026-11-09 every file on
  the boot drive (a 2 GB USB stick inside the box) read without error, and
  UR's checksum lists on it (/pc.md5sums, which covers URControl, URLLControl
  and the joints' settings files; /joint_firmware.md5sums; /common.md5sums;
  the board's) all match, except files meant to change
  (touch_calibration.properties, the GUI's config.properties) or deleted
  (two other models' help, old installers). urcontrol.conf,
  robot_calibration_summary.txt and safety.conf (same lines; same safety
  CRCs at start-up) are as on 2026-09-29. So a re-image would not help, as it
  did a CB3 with the same "timed out" on UR's forum (topic 15309: a
  corrupted drive). Also not: the arm cable's plug at the box (reseated
  2026-11-09), the wall socket, the internal link to the safety board (eth0,
  no errors). "urEthernetDriver: Lost eth package" (thousands per start),
  C4A88, C4A91 and C5A1-C5A4 ("heavy processor load") also appear on the
  days it worked: normal for this arm. While booting, the joints' reported
  currents (±66 A) and temperatures are not to be believed.
- **Seen again 2026-11-12** (controller clock about 7 h ahead of the Mac):
  the arm had run all day; at 21:27 a pick was stopped and the e-stop pressed,
  the controller was shut down from its power button and later the mains.
  From then every ON stopped at "powering on", all six joints booting (after
  the mains-off start not even reporting their angles), the 48 V supply at
  45.7-45.8 V (46.7-46.8 V while it ran that morning), and the controller
  had started with `external.urp` loaded (the previous owner's program, set
  to load at start-up; "No program loaded" during the day).
- **Workaround that worked (2026-11-12): load the installation again, then
  power on.** On the pendant: File > Load Installation > `default.installation`
  (closing `external.urp` without saving), then on the initialization screen
  ON, and START when it offers it: the joints booted and the arm was back. On
  2026-11-05 the same reload got the joints answering, though they stayed
  booting then. `external.urp` itself uses installation "default", so it is
  the reload after start-up that counts, not other settings. Try this first,
  before anything inside the box; `tools/ur5/check_connection.py` (from
  Terminal) says whether the joints answer and boot.
- **Workaround otherwise:** none that helped: OFF/ON, controller restarts, a mains-off
  cold start, another wall socket, the cable reseat. Next: swap in the safety
  control board from one of the two spare CB3 boxes (service manual 3.2.3:
  low voltage only, after checking the spare starts on its own and its board
  is not newer than the 3.15.6 software), then the 48 V supply (3.2.5: mains
  wiring). Afterwards test the emergency stop before the arm is used. Moving
  this box's drive into a spare box and the arm with it would test the whole
  box at once. `tools/ur5/check_connection.py` prints the 48 V supply and
  each joint's state while the arm powers up. The logs above are read over
  the controller's SSH (as root), never written there. Change UR's factory
  root password on any arm on a shared network.
- **Status:** open, with the workaround above. Why it ran again from the
  morning of 2026-11-12 after a week stuck was not recorded; why a reload
  frees it is not known. To try: stop `external.urp` loading at start-up
  (the installation's default-program setting on the pendant), and whether
  the e-stop alone, without the restarts, leaves it stuck.

### The arm has no kinematic calibration file
- **Seen:** 2026-11-05: URControl warns at every start that
  /root/.urcontrol/calibration.conf is missing, and did on 2026-09-29 too. The
  drive looks re-imaged with 3.15.6 around 2022-01 (file dates); the file was
  not put back.
- **Cause:** the controller runs on the nominal UR5 geometry, so its poses may
  be a millimetre or two off the arm's own. kinematics.py uses the same
  nominal values, which is why the two agreed to 0.0 mm.
- **Workaround:** none yet. The file comes from UR (by the arm's serial) or
  from a backup of the original drive.
- **Status:** open.

### macOS blocks the arm until the app has Local Network
- **Seen:** 2026-11-05, first cable to the arm: a gigabit link, the arm's ARP
  requests arriving (`tcpdump` on the adapter), but every ping from VS Code's
  terminal failed at once with `sendto: No route to host`, and the adapter's
  sent-packet count did not move. The same ping from Terminal got replies.
  Other devices on the Wi-Fi failed the same way, the Wi-Fi router did not.
- **Cause:** macOS's Local Network privacy. An app without it cannot reach any
  address on a local network, and the error is "No route to host", not a
  prompt. The arm's address and the adapter were never at fault.
- **Workaround:** System Settings > Privacy & Security > Local Network, on for
  the app the tools run from, then quit and reopen that app. Terminal worked;
  VS Code's terminal was still blocked after it was turned on and VS Code
  relaunched (2026-11-05), so run the arm's tools from Terminal
  (`open -a Terminal script.command` runs a script there from any app). Also:
  with no router, the pendant shows "Not connected to network!" (it cannot
  reach its DNS server, 8.8.8.8); the link works anyway.
- **Seen again 2026-11-12:** in the morning ping and nc reached the arm from
  VS Code (Python did not); after the USB-Ethernet adapter was unplugged and
  plugged back in, nothing from VS Code did — ARP incomplete, no packet in —
  while Terminal reached the arm at once. It looks exactly like a dead link:
  test from Terminal before touching any cable.
- **Status:** open: a step for each new Mac and each new app; VS Code unsolved.

### ur_rtde has no macOS build
- **Seen:** 2026-11-02, `pip install ur_rtde` 1.6.5: PyPI has wheels for Linux
  and Windows only, so on a Mac it builds from source, and that build failed
  before compiling anything. The Mac's Command Line Tools ship the macOS 27 SDK,
  which Xcode's linker cannot read (`ld: tapi error: malformed file ...
  MacOSX27.0.sdk/usr/lib/libSystem.B.tbd: unknown architecture arm64e.x1`).
  A plain C hello-world failed the same way.
- **Cause:** not ur_rtde. On this Mac, the Command Line Tools' SDK is newer
  than Xcode's linker.
- **Workaround:** build against Xcode's own SDK, in the Python that runs the
  arm (the `mrcf-ur5` conda env on the author's Mac):
  ```bash
  brew install boost cmake
  SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install ur_rtde==1.6.5 --no-binary ur_rtde
  ```
  The result links Homebrew's Boost: after `brew upgrade boost`, build it again.
- **Status:** open, until ur_rtde ships macOS wheels or the two SDKs agree.

### The Hand-E's numbers are from its datasheet, not measured
- **Seen:** 2026-11-02, from a photo of the rig: a Robotiq Hand-E (it was
  first assumed to be a 2F-85).
- **Cause:** written before the arm was on the network.
- **Workaround:** `GRIPPER = None` in `robot_config.py` (or `make_env(gripper=None)`)
  runs the arm without it: everything but grasping works, and pick and place
  are refused before the arm moves.
- **Measured 2026-11-12:** its URCap answers on 63352 (a SET with a bare
  "ack", no newline); `ROBOTIQ_POS_OPEN` 3, `_CLOSED` 248; the fingertips 52 mm
  apart fully open; flange face to fingertip ends 157 mm (so the grasp point's
  0.146 m and the tips' 11 mm stand); the payload 1.1 kg on the pendant, and
  sent by every connect with its centre of gravity (`PAYLOAD_COG_M`).
- **Status:** open for what is still the datasheet's or a guess: the centre of
  gravity (5.5 cm out, estimated; Robotiq's manual has it). The axis the
  fingers open along as mounted (`JAW_SLIDE_AXIS`) was a guess a quarter turn
  wrong; corrected and checked on the rig 2026-11-20
  (`tools/check_jaw_direction.py`, `rig/ur5/checked_facts.json`).
