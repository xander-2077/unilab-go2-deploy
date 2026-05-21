# UniLab Go2 Deployment

This directory contains lightweight deployment scripts for running UniLab policies on a Unitree Go2.

## Environment

Use the wrapper so `uv` loads `.env` and the matching CycloneDDS libraries:

```bash
./run.sh
```

Equivalent explicit form:

```bash
/home/unitree/.local/bin/uv run --env-file .env python deploy.py
```

## Models

Policy files are expected under `models/<TaskName>/policy.onnx` or checkpoint
subdirectories such as `models/Go2FootStand/<CheckpointName>/policy.onnx`, for example:

- `models/Go2FootStand/policy.onnx`
- `models/Go2FootStand/2026-05-21_14-16-56_mujoco/policy.onnx`
- `models/Go2JoystickFlat/policy.onnx`
- `models/Go2HandStand/policy.onnx`

Model files are ignored by git because they are large deployment artifacts.

## Go2FootStand

Default launch:

```bash
./run.sh
```

By default deployment uses the latest timestamped checkpoint under `models/Go2FootStand`.
You can list or select checkpoints with:

```bash
./run.sh python deploy.py --list-policy-ckpts
./run.sh python deploy.py --policy-ckpt 2026-05-21_14-16-56_mujoco
./run.sh python deploy.py --policy-ckpt latest
```

The direct path form is still supported and overrides `--policy-ckpt`:

```bash
./run.sh python deploy.py --policy models/Go2FootStand/2026-05-21_14-16-56_mujoco/policy.onnx
```

The deployment adapter matches the UniLab `Go2FootStand` actor ABI:

- 45 values per frame: local linear velocity, gyro, local gravity, joint position delta, joint velocity, previous action.
- 15-frame observation history, flattened to 675 values.
- 12 actions in Unitree real/control order, clipped to `[-1, 1]`, integrated into motor targets with `action_scale=0.3`.
- By default FootStand actions are not filtered or delta-limited, matching UniLab `Go2FootStandTask.apply_action()`.

The Go2 low-state stream does not provide base linear velocity, so deployment fills that 3-value ABI slot with zeros.

The first L1 stand-up phase follows the Unitree/mjlab Go2 `FixStand` trajectory:

- current joint q -> `[0.0, 1.36, -2.65]` for each leg.
- `[0.0, 1.36, -2.65]` -> `[0.0, 0.8, -1.5]` for each leg.
- stand gains use per-joint real-order `kp=[60,80,80] * 4`, `kd=[5,4,4] * 4`.
- the deployment torque limiter is temporarily disabled during this phase so the lowcmd target q matches the FixStand waypoints; it is restored before policy control.

Useful options:

```bash
./run.sh python deploy.py --debug-command
./run.sh python deploy.py --dry-run 8
./run.sh python deploy.py --torque-limit-scale 0.65 --startup-torque-limit-scale 0.45 --debug-command
./run.sh python deploy.py --stand-seconds 2
./run.sh python deploy.py --action-filter-alpha 0.35 --max-action-delta 0.2
```

By default, real-robot commands are protected by a PD torque limiter in `robot.py`.
It estimates `kp * (q_cmd - q_measured) - kd * dq` before each lowcmd publish and pulls `q_cmd` back when the estimate exceeds the active scaled URDF torque limits.
The default policy-control limit ramps from `0.45 *` to `0.65 *` the URDF limits over the first 3 seconds, then holds `0.65 *` for the task.
If the robot still overloads, lower `--torque-limit-scale`; if it cannot rise but `tau_hits` stays high, raise it gradually while watching motor temperature and error lights.

## Go2JoystickFlat

Run the older joystick policy explicitly:

```bash
./run.sh python deploy.py --policy models/Go2JoystickFlat/policy.onnx
```

Useful options:

```bash
./run.sh python deploy.py --policy models/Go2JoystickFlat/policy.onnx --debug-command
./run.sh python deploy.py --policy models/Go2JoystickFlat/policy.onnx --dry-run 8
./run.sh python deploy.py --policy models/Go2JoystickFlat/policy.onnx --fixed-command --command 0.5 0.0 0.0
```

The stand-up phase interpolates from the current joint offsets to the standing pose, then policy control starts on the second L1 press. Runtime gains are aligned with UniLab Go2 defaults (`Kp=35.0`, `Kd=0.5`).

## Safety Notes

- First L1: interpolated stand-up.
- Second L1: start policy control.
- First L2 during policy control: stop policy control and hold the current command.
- Release L2, then press L2 again: set all `kp/kd` to zero and exit.
- `rt/lowstate.wireless_remote` is used for joystick axes/buttons on this Go2.
