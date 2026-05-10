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

Policy files are expected under `models/<TaskName>/policy.onnx`, for example:

- `models/Go2JoystickFlat/policy.onnx`
- `models/Go2HandStand/policy.onnx`

Model files are ignored by git because they are large deployment artifacts.

## Go2JoystickFlat

Default launch:

```bash
./run.sh
```

Useful options:

```bash
./run.sh python deploy.py --debug-command
./run.sh python deploy.py --dry-run 8
./run.sh python deploy.py --fixed-command --command 0.5 0.0 0.0
```

The stand-up phase interpolates from the current joint offsets to the standing pose, then policy control starts on the second L1 press. Runtime gains are aligned with UniLab Go2JoystickFlat defaults (`Kp=35.0`, `Kd=0.5`).

## Safety Notes

- First L1: interpolated stand-up.
- Second L1: start policy control.
- L2: stop policy control and relax.
- `rt/lowstate.wireless_remote` is used for joystick axes/buttons on this Go2.
