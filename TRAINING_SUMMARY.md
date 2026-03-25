# MATH4025-MINECRAFT SAC Training Summary

## Project Overview

Train a Soft Actor-Critic (SAC) reinforcement learning agent to chop wood in Minecraft using the MineRL framework. The environment is a custom `GatherWood-v0` built on MineRL's BASALT `HumanControlEnvSpec`.

---

## Architecture

### Wrapper Stack (innermost → outermost)
```
GatherWoodEnvironment (MineRL)
  → LogRewardWrapper       (+100 per log, 1000 step timeout, 2s sleep between resets)
  → StickyAttackWrapper    (holds attack for 15 ticks)
  → WoodDetectionRewardWrapper  (HSV pixel detection, all rewards currently 0)
  → [RenderWrapper]        (optional, calls env.render() each step)
  → PovImageWrapper        (64x64 RGB → (C,H,W) uint8)
  → ActionWrapper          (7-dim continuous [-1,1] → MineRL action dict)
```

### Action Space (7-dim)
```
[0] camera pitch  — clamped to [-0.3, 1.0] * 10° (limits upward look)
[1] camera yaw    — [-10°, +10°]
[2] forward       — > 0 → 1
[3] back          — > 0 → 1
[4] left          — > 0 → 1
[5] right         — > 0 → 1
[6] attack        — > 0 → 1
```

### SAC Hyperparameters (current)
```python
learning_rate    = 1e-5
gamma            = 0.99
tau              = 5e-3
train_freq       = 4
gradient_steps   = 1
learning_starts  = 5000
buffer_size      = 100_000
batch_size       = 128
replay_buffer    = NStepReplayBuffer(n_steps=50, gamma=0.99)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `environment/wood_environment.py` | Environment wrappers and GatherWood-v0 spec |
| `model/sac/run.py` | SAC training loop |
| `model/sac/pretrain.py` | Behavioral Cloning pretraining from MineRL dataset |
| `model/sac/replay_buffer.py` | N-step replay buffer |
| `model/sac/callbacks.py` | RewardPlotCallback — saves CSV after every episode |
| `model/sac/bc_callback.py` | BCRegularizationCallback — implemented but abandoned |
| `model/sac/eval_checkpoints.py` | Evaluate checkpoint zips, generate reward plots |
| `model/main.py` | Entry point, CLI args |
| `Makefile` | Convenience targets |

---

## Reward Structure Evolution

| Stage | Config | Result |
|-------|--------|--------|
| Initial | LOOK=0.01, MINE=0.05, APPROACH=0.02, LOG=+1 | Agent stared at wood. Episode reward=78 (8k steps = 8000×0.01), then collapsed |
| Remove look | LOOK=0, MINE=0.05, APPROACH=0.02, LOG=+1 | Improvement but approach reward noisy |
| Remove approach | LOOK=0, MINE=0.05, APPROACH=0, LOG=+1 | Cleaner signal |
| Increase log | LOG=+100 | Better sparse signal |
| Fully sparse | All shape=0, LOG=+100 | Current state |

**Current rewards:**
- `+100` per log collected (inventory change)
- `+0` everything else

---

## Training Runs

### Run 1 — Dense rewards, no pretrain
- **Result:** reward=78 at 8k (staring exploit), collapsed to 0-16 after

### Run 2 — Sparse rewards + BC pretrained + diamond axe
- **Result:** sporadic 100-200 at 40k-96k steps — actual log collection

### Run 3 — Pretrained + 500 step timeout + gradient_steps=4
- **Result:** first success at 10,891, two more at 11,436/13,618, then all zeros — catastrophic forgetting

### Run 4 — Pretrained + 500 step timeout + gradient_steps=1
- **Result:** all zeros to 32k — 500 steps not enough time

### Run 5 — Pretrained + 1000 step timeout + n_steps=50
- **Result:** successes at 7,259 and 7,340, went cold at 12k — catastrophic forgetting

### Run 6 — BC Regularization (lambda=0.5)
- **Result:** 20k all zeros — regularization too strong, blocked exploration

### Run 7 — lr=1e-5 + learning_starts=5000 + pretrained
- **Result:** all episodes timed out — pretrained policy from older Minecraft version doesn't transfer

### Run 8 — lr=1e-5 + learning_starts=5000 + no pretrained (current)
- **Result:** agent IS attacking wood (wood ratio 0.78-0.93) but timing out before breaking block

---

## Key Problems and Solutions

### Catastrophic Forgetting
**Problem:** SAC overwrites BC pretrained weights within 10-15k steps
**Attempts:** lower LR (1e-4 → 1e-5), gradient_steps=4 → 1, BC regularization
**Status:** Partially solved with gradient_steps=1 + lr=1e-5 but still occurring

### Upward Pitch Exploitation
**Problem:** Agent learned to always look up at treetops after 40k steps
**Fix:** Clamped upward camera pitch to 30% of max in `ActionWrapper`

### Malmo Reset Crashes
**Problem:** Rapid resets (500/1000 step timeout) caused Java process TimeoutError
**Fix:** `time.sleep(2)` in `LogRewardWrapper.reset()`

### BC Pretraining Distribution Mismatch
**Problem:** MineRLTreechop-v0 data is from an older Minecraft version — different textures, lighting, world generation. Pretrained policy fails to collect logs in 1000 steps on HPC.
**Status:** Pretraining may be net negative

### HSV Detection Noise
**Problem:** Approach reward fired when agent spun to bring tree pixels into frame
**Fix:** Removed approach and look rewards entirely

---

## N-Step Replay Buffer Math

Standard TD update (single step):
```
Q(s_t, a_t) ← r_t + γ · V(s_{t+1})
```
Only the transition at the reward step gets updated.

N-step return (n=50):
```
R_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ⁴⁹r_{t+49}
```
A +100 reward at step 50 contributes `0.99^49 × 100 = 61` back to step 0. **50 transitions get credited simultaneously** from a single log collection.

---

## HPC Setup

```bash
# Singularity container
PATH_TO_MINERL_SANDBOX="/home/liu.sto/minerl-3.10-sandbox"

# Bind mount: /home/liu.sto (host) → /home (container)
singularity exec --nv -w \
  --bind /home/liu.sto:/home \
  --env PYTHONPATH=/home/MATH4025-Minecraft-Project \
  $PATH_TO_MINERL_SANDBOX \
  /home/singularity-minerl/setupvgl.sh \
  /opt/conda/envs/minerl/bin/python -m model.main --mode sac \
  --timesteps 200000 \
  --checkpoint-out /home/artifacts/sac_from_scratch
```

**Path mapping:**
| Host | Container |
|------|-----------|
| `/home/liu.sto/artifacts/` | `/home/artifacts/` |
| `/home/liu.sto/MATH4025-Minecraft-Project/` | `/home/MATH4025-Minecraft-Project/` |

---

## Current Status

The agent is **finding and attacking wood** (wood ratio 0.78-0.93 confirmed in logs) but timing out before breaking the block. Next steps:

1. Re-enable `MINE_REWARD` (+0.5) to incentivize sustained attacking
2. Increase `sticky_ticks` from 15 to 30 to commit longer to attacks
3. Continue no-pretrain run — agent is behaviorally close to collecting first log

---

## Make Commands

```bash
# Train from scratch
make sac TIMESTEPS=200000 CHECKPOINT_OUT=artifacts/sac_v6

# Train with pretrained weights
make sac PRETRAINED=artifacts/sac_pretrained.zip TIMESTEPS=200000 CHECKPOINT_OUT=artifacts/sac_v6

# Resume from checkpoint
make sac CHECKPOINT=artifacts/sac_v6/sac_wood_50000_steps.zip TIMESTEPS=200000 CHECKPOINT_OUT=artifacts/sac_v6

# Evaluate a checkpoint
make eval-checkpoints EVAL_CHECKPOINT=artifacts/sac_v6/sac_wood_100000_steps.zip EVAL_EPISODES=5

# Watch agent play
make sac CHECKPOINT=artifacts/sac_v6/sac_wood_100000_steps.zip RENDER=1 TIMESTEPS=1000
```
