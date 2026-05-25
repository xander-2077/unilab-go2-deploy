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
- `models/Go2FootStand/2026-05-24_05-19-37_mujoco/policy.onnx`
- `models/Go2JoystickFlat/policy.onnx`
- `models/walk_these_ways/body_latest.jit`
- `models/walk_these_ways/adaptation_module_latest.jit`
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
./run.sh python deploy.py --policy-ckpt 2026-05-24_05-19-37_mujoco
./run.sh python deploy.py --policy-ckpt latest
```

The direct path form is still supported and overrides `--policy-ckpt`:

```bash
./run.sh python deploy.py --policy models/Go2FootStand/2026-05-24_05-19-37_mujoco/policy.onnx
```

If launching from the repository root instead of inside `deployment/`, the
same checkpoint can also be passed as `deployment/models/Go2FootStand/2026-05-24_05-19-37_mujoco/policy.onnx`.

The runtime state machine loads both policies:

- after the first L1 stand-up, it starts `models/walk_these_ways` for joystick velocity tracking.
- after stand-up finishes, it holds the final FixStand posture for 2 seconds before handing control to the velocity policy; tune this with `--post-stand-delay`.
- after L1 is released, press L1 again to switch once into the selected Go2FootStand checkpoint.
- before switching to FootStand, it runs the velocity policy with a zero command for 0.75 seconds; tune this with `--pre-footstand-hold-seconds`.
- then it interpolates the joints to the shared UniLab Go2FootStand start pose and only starts FootStand after measured joint error and velocity are within tolerance. In sim order `FL,FR,RL,RR`, this pose is `[0,0.8,-1.5, 0,0.8,-1.5, 0,1.0,-1.5, 0,1.0,-1.5]`.
- `--velocity-policy` can override the joystick policy path; use `models/Go2JoystickFlat/policy.onnx` here if you want the ONNX joystick policy.
- `--policy` or `--policy-ckpt` selects the FootStand policy.
- after switching to FootStand, the logger writes observation, action, target,
  LowCmd, and torque-limiter fields to
  `logs/footstand_<ckpt>_<YYYYmmdd_HHMMSS_BJT>.csv`; use `--log-dir` to change
  the output directory.

The default `walk_these_ways` velocity adapter matches the legacy TorchScript ABI:

- `adaptation_module_latest.jit` consumes 30 frames x 70 values and outputs the 2-value latent.
- `body_latest.jit` consumes the flattened 2100-value history plus latent and outputs 12 actions.
- each 70-value frame is gravity, 15 command values, joint position delta, scaled joint velocity, current action, previous action, and 4 gait clock values.

The deployment adapter matches the UniLab `Go2FootStand` actor ABI:

- Current distilled FootStand checkpoints consume 42 values per frame: gyro, local gravity, joint position delta, joint velocity, previous action.
- The 15-frame observation history is flattened to 630 values for the distilled actor, matching `2026-05-24_05-19-37_mujoco`.
- Legacy 675-value FootStand checkpoints are still supported by selecting the 45-value frame adapter, which keeps a zero-filled local linear velocity slot.
- 12 actions in Unitree real/control order, clipped to `[-1, 1]`, integrated into motor targets with `action_scale=0.3`.
- FootStand actions are applied immediately, matching the `simulate_action_latency=false` default used by `train_rsl_rl_student_finetune.py` for this run.
- By default FootStand actions are not filtered or delta-limited, matching UniLab `Go2FootStandTask.apply_action()`.

The first L1 stand-up phase follows the Unitree/mjlab Go2 `FixStand` trajectory:

- current joint q -> `[0.0, 1.36, -2.65]` for each leg.
- `[0.0, 1.36, -2.65]` -> `[0.0, 0.8, -1.5]` for each leg.
- stand gains use per-joint real-order `kp=[60,80,80] * 4`, `kd=[5,4,4] * 4`.
- the deployment torque limiter is temporarily disabled during this phase so the lowcmd target q matches the FixStand waypoints; it is restored before policy control.

Useful options:

```bash
./run.sh python deploy.py --debug-command
./run.sh python deploy.py --dry-run 8
./run.sh python deploy.py --torque-limit-scale 1.0 --startup-torque-limit-scale 1.0 --startup-torque-ramp-seconds 0 --debug-command
./run.sh python deploy.py --stand-seconds 2
./run.sh python deploy.py --post-stand-delay 2.0
./run.sh python deploy.py --pre-footstand-hold-seconds 0.75
./run.sh python deploy.py --pre-footstand-pose-seconds 1.0 --pre-footstand-pose-tolerance 0.16 --pre-footstand-dq-tolerance 1.0
./run.sh python deploy.py --relax-hold-seconds 5.0
./run.sh python deploy.py --action-filter-alpha 0.35 --max-action-delta 0.2
```

By default, real-robot commands are protected by a PD torque limiter in `robot.py`.
It estimates `kp * (q_cmd - q_measured) - kd * dq` before each lowcmd publish and pulls `q_cmd` back when the estimate exceeds the active URDF torque limits.
The default policy-control limit uses `--torque-limit-scale 1.0` with no startup ramp, so the deploy-side limiter no longer applies the old `0.65` scale.
If the robot overloads, lower `--torque-limit-scale`; if it cannot rise but `tau_hits` stays high, raise it gradually while watching motor temperature and error lights.

When running against `unitree_mujoco` with `--sim`, the deploy-side torque limiter
is disabled by default so the LowCmd stream is the same position-control pipeline
used by the MuJoCo bridge: absolute `q`, `dq=0`, `tau=0`, and per-motor `kp/kd`.
Use `--enable-sim-torque-limit` only when deliberately testing the deploy
limiter in simulation.

## Go2JoystickFlat

Use Go2JoystickFlat as the optional ONNX velocity-tracking stage:

```bash
./run.sh python deploy.py --velocity-policy models/Go2JoystickFlat/policy.onnx --debug-command
```

Useful options:

```bash
./run.sh python deploy.py --velocity-policy models/Go2JoystickFlat/policy.onnx --dry-run 8
./run.sh python deploy.py --velocity-policy models/Go2JoystickFlat/policy.onnx --fixed-command --command 0.5 0.0 0.0
```

The stand-up phase interpolates from the current joint offsets to the standing pose, then policy control starts on the second L1 press. Runtime gains are aligned with UniLab Go2 defaults (`Kp=35.0`, `Kd=0.5`).

## Safety Notes

- First L1: interpolated stand-up.
- After stand-up: hold the final stand pose for 2 seconds, then start joystick velocity tracking.
- Release L1, then press L1 again: switch to FootStand policy.
- The FootStand switch first holds zero velocity, moves joints to the UniLab reset pose, and aborts instead of running the policy if the pose is not confirmed.
- First L2 during policy control: stop policy control and set `kp=0,kd=10`.
- Release L2, then press L2 again: set all `kp/kd` to zero and keep publishing relaxed lowcmds.
- By default the relaxed lowcmd keepalive runs until Ctrl+C; use `--relax-hold-seconds N` to exit automatically after `N` seconds.
- `rt/lowstate.wireless_remote` is used for joystick axes/buttons on this Go2.
