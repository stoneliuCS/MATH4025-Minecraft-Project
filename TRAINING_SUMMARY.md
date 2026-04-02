# SAC Wood-Chopping Agent — Development Summary

## Overview

A Soft Actor-Critic (SAC) reinforcement learning agent trained to chop wood in Minecraft 1.16 using the MineRL framework on an HPC cluster (SLURM + Singularity + GPU). The agent observes 64×64 RGB POV frames and outputs continuous actions mapped to movement, camera, and attack.

---

## Environment (`environment/wood_env2.py`, `environment/ray.py`)

### Base Environment
- `GatherWoodEnvironment` — custom `HumanControlEnvSpec` registered as `GatherWood-v0`
- Spawns in forest biome with 64 diamond axes
- Observables: POV camera, current location, life stats, `ObserveFromFullStats("mine_block")`, `ObservationFromRay`

### Wrapper Chain (outermost to innermost)
```
Monitor (SB3)
GymV21CompatibilityV0 (shimmy — old gym → gymnasium boundary)
ActionWrapper          (numpy [8] → MineRL dict)
PovImageWrapper        (dict obs → (3, 64, 64) uint8, channel-first)
RenderWrapper          (optional, calls env.render() each step)
StickyAttackWrapper    (holds attack=1 for 5 ticks after each attack action)
MineBlockRewardWrapper (custom reward shaping)
RobustResetWrapper     (crash recovery — restarts Minecraft on timeout/error)
MineRL base env
```

### Action Space (8-dim continuous, [-1, 1])
| Index | Action | Mapping |
|---|---|---|
| 0 | camera pitch | × CAMERA_MAX_ANGLE |
| 1 | camera yaw | × CAMERA_MAX_ANGLE |
| 2 | forward | > 0 → 1 |
| 3 | back | > 0 → 1 |
| 4 | left | > 0 → 1 |
| 5 | right | > 0 → 1 |
| 6 | attack | > 0.25 → 1 |
| 7 | jump | > 0.95 → 1 |

### Environment Parameters (current)
```python
MAX_EPISODE_STEPS = 2000      # was 1000
CAMERA_MAX_ANGLE  = 10.0      # was 5.0 — larger angle = faster scanning/aiming
FRAME_SIZE        = 64
ACTION_DIM        = 8
```

---

## Reward Function (`MineBlockRewardWrapper`)

| Event | Reward |
|---|---|
| Break any log (12 types incl. stripped) | +2.0 per block |
| Attack while looking at log, in range | +0.2 per step |
| Break any leaf block (6 types) | +0.0001 per block |
| Break dirt | -0.002 per block |
| Break grass block | -0.002 per block |
| Per timestep | -0.0002 |

### Key implementation details
- **Cumulative delta tracking**: `ObserveFromFullStats("mine_block")` returns lifetime cumulative counts. The wrapper diffs against `prev_mine_counts` each step.
- **Corrupt spike guard**: if total delta across all block types > 10 in one step, it's treated as a stats-dump artifact (ObservationFromFullStats sometimes delivers all lifetime counts at once on episode start). Counts are absorbed silently without awarding reward.
- **`_seeded` flag**: `prev_mine_counts` is seeded from the first step obs (not reset obs) to avoid a spike from the zero-baseline.
- **Delta cap**: max 5 blocks per type per step even after the spike guard, as a secondary safety net.

### Ray Observation (`environment/ray.py`)
- `ObservationFromRay` wraps Malmo's `<ObservationFromRay/>` XML
- Provides per-step `LineOfSight`: block type flags (12 log types), `in_range`, `distance`, xyz
- Also carries `mine_block` cumulative counts (20 block types: 12 logs + 6 leaves + dirt + grass)
- Used for `ATTACK_LOG_REWARD` shaping: fires when `looking_at_log AND in_range AND attack_active`

---

## SAC Hyperparameters (`model/sac/run.py`)

```python
buffer_size    = 500_000
batch_size     = 512
learning_rate  = 3e-4   (reduced to 1e-4 on checkpoint resume to prevent NaN)
gamma          = 0.99
tau            = 5e-3
train_freq     = 4
gradient_steps = 8
learning_starts = 500
replay_buffer  = NStepReplayBuffer(n_steps=10, gamma=0.99)
policy         = "CnnPolicy"
```

### NaN Prevention (added to checkpoint resume)
```python
# Abort if checkpoint has NaN weights
nan_params = [n for n, p in model.policy.named_parameters() if torch.isnan(p).any()]
if nan_params:
    raise ValueError(f"Checkpoint has NaN weights in: {nan_params[:5]} — try an earlier checkpoint")

# Reduce LR to stabilise
for param_group in model.policy.optimizer.param_groups:
    param_group["lr"] = 1e-4
```

---

## Behavioural Cloning Pretraining (`model/sac/pretrain.py`)

- Loads MineRL `MineRLTreechop-v0` human demonstration dataset
- Trains SAC actor via negative log-likelihood on demo actions (BC loss)
- 8-dim action space: `[pitch, yaw, forward, back, left, right, attack, jump]`
  - Jump column is all -0.5 (not demonstrated by humans)
- Output: `artifacts/sac_pretrained.zip`

---

## Issues Encountered and Fixes

| Issue | Fix |
|---|---|
| SLURM job cancelled (Malmo TimeoutError) | Added `RobustResetWrapper` with 3-retry crash recovery |
| `AssertionError: Expected env to be gymnasium.Env` | Moved `GymV21CompatibilityV0` to outermost position (just inside Monitor) |
| `seed` kwarg error on reset | Strip `seed`/`options` kwargs in `RobustResetWrapper.reset()` |
| `cv2.resize` crash on reset | `GymV21CompatibilityV0` was too low — reset returned `(obs, info)` tuple |
| `mine_block` always 0 | Added `handlers.ObserveFromFullStats("mine_block")` to `create_observables()` |
| Wrong obs path | `obs["ray"]["mine_block"]` → `obs["ray"]["ray_data"]["mine_block"]` |
| Corrupt reward spike (+60 in one step) | `total_delta > 10` guard absorbs stats-dump events silently |
| Action space mismatch (7 vs 8 dim) in BC pretrain | Added jump column to pretrain action stack |
| NaN in actor weights | Checkpoint NaN check on resume + reduce LR to 1e-4 |
| Monitor CSV overwritten on resume | Timestamped monitor filename: `monitor_{int(time.time())}` |
| Minecraft client lag after ~100 steps | Caused by per-step debug `print()` statements blocking stdout I/O over SSH |
| X display crash → GLFW init failure | Use `-noreset` flag with Xvfb in SLURM script |

---

## Training Results (`sac_ray_tracing_rewards_fine_tuned_v6`)

Episode reward baseline: `-0.4` (2000 steps × 0.0002 time penalty, agent does nothing)

| Timestep | Peak episode reward | Est. logs chopped |
|---|---|---|
| 128k | 15.2 | ~4 |
| 316k | 33.0 | ~10 |
| 360k | 71.4 | ~21 |
| 426k | 86.0 | ~25 |
| 484k | 119.2 | ~35 |
| 532k | 135.2 | ~40 |

Theoretical maximum: ~70–100 logs per episode (~250–340 reward), assuming efficient tree-to-tree navigation.

Learning breakthrough occurred at ~316k steps following changes:
- `CAMERA_MAX_ANGLE` 5° → 10°
- `MAX_EPISODE_STEPS` 1000 → 2000
- `TIME_PENALTY` 0.0005 → 0.0002
- `LOG_REWARD` 1.0 → 2.0
- `ATTACK_LOG_REWARD` 0.1 → 0.2

---

## SLURM / Infrastructure Notes

- Xvfb display number: `$((SLURM_JOB_ID % 100 + 1))` — avoids conflicts between concurrent jobs
- Xvfb flags: `-noreset` keeps display alive when Minecraft crashes and restarts
- Resume from checkpoint: `--checkpoint path/to/sac_wood_XXXXXX_steps.zip`
- Checkpoint auto-saves every 10k steps to the same directory as the loaded checkpoint
- `checkpoint_out` is auto-derived from the checkpoint path on resume

### SLURM script pattern
```bash
DISPLAY_NUM=$((SLURM_JOB_ID % 100 + 1))
rm -f /tmp/.X${DISPLAY_NUM}-lock /tmp/.X11-unix/X${DISPLAY_NUM}
Xvfb :${DISPLAY_NUM} -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 3

singularity exec --nv -w \
    --bind /home/liu.sto:/home \
    --env PYTHONPATH=/home/MATH4025-Minecraft-Project \
    --env DISPLAY=:${DISPLAY_NUM} \
    $PATH_TO_MINERL_SANDBOX \
    /home/singularity-minerl/setupvgl.sh \
    /opt/conda/envs/minerl/bin/python -m model.main --mode sac \
    --timesteps 1000000 \
    --checkpoint /home/artifacts/.../sac_wood_XXXXXX_steps.zip

kill $XVFB_PID 2>/dev/null
```

---

## Plotting

Monitor files from SB3 are saved as `monitor_{timestamp}.monitor.csv` in the checkpoint directory. To regenerate the reward plot from monitor files:

```bash
python -m model.sac.plot_monitor \
    --monitor-dir data/results/sac_ray_tracing_rewards_fine_tuned_v6 \
    --out data/results/reward_monitor_plot.png
```
