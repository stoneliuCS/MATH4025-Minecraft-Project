# SAC (Soft Actor-Critic) Implementation Plan

## Context

The project currently uses DQN with 4 discrete movement actions for a boxed navigation task. To enable wood-gathering (which requires camera control + attack), we need to implement SAC from Haarnoja et al. 2018 with a continuous action space. This is also distinct from the PPO implementation being done by a teammate.

## Action Space Design

5D continuous vector (tanh-squashed to [-1, 1]):

| Dim | Meaning | Conversion to MineRL |
|-----|---------|---------------------|
| 0 | forward/back | > 0 → forward=1, < 0 → back=1 |
| 1 | left/right | > 0 → right=1, < 0 → left=1 |
| 2 | camera pitch | scaled to [-10, 10] degrees |
| 3 | camera yaw | scaled to [-10, 10] degrees |
| 4 | attack | > 0 → attack=1 |

## Files to Create

### 1. `model/sac/__init__.py` — empty package marker

### 2. `model/sac/networks.py` — Neural networks

- **CNNEncoder**: Reuses Nature DQN conv layers (4×64×64 → 1024 features)
  - Conv2d(4→32, k=8, s=4) → Conv2d(32→64, k=4, s=2) → Conv2d(64→64, k=3, s=1)
- **GaussianActor**: encoder → FC(1024→256) → FC(256→256) → mean_head + log_std_head → tanh squashing with reparameterization trick
- **QNetwork**: encoder → concat(state_features, action) → FC(1029→256) → FC(256→256) → FC(256→1)
- **TwinQNetwork**: wraps two independent QNetworks (clipped double-Q)

### 3. `model/sac/replay_buffer.py` — Replay buffer

- Same circular buffer structure as `model/deep-q-learning/replay_buffer.py`
- Only change: actions stored as float32 (5,) instead of int64 scalar

### 4. `model/sac/wrappers.py` — Action wrapper + observation wrappers

- **ContinuousActionWrapper**: converts 5D float → MineRL dict, ports reward shaping from `environment/restricted_wrapper.py`
- Copy `GrayscaleWrapper` and `FrameStackWrapper` from `model/deep-q-learning/wrappers.py` (can't import directly due to hyphenated directory name)

### 5. `model/sac/train.py` — SAC agent + training/eval loops

**SACAgent class** with:
- Actor (GaussianActor) + twin critics + target critics
- Automatic entropy tuning (learnable log_alpha, target entropy = -5)
- Polyak soft target updates (τ=0.005)
- `select_action(state, evaluate)` — stochastic for training, deterministic mean for eval
- `update(batch)` — critic loss (MSE on soft Bellman target), actor loss (maximize Q - α·log π), alpha loss

**Hyperparameters:**
- LR: 3e-4 (actor, critic, alpha)
- γ=0.99, τ=0.005, batch_size=256
- Replay buffer: 100k, warmup: 5k steps
- Episodes: 500, max steps: 2000
- Gradient clipping: max_norm=10.0

**Training loop**: warmup with random actions → epsilon-free (SAC explores via entropy) → log metrics per episode → checkpoint every 25 episodes to `artifacts/sac_model.pt`

**Eval function**: load checkpoint, run deterministic policy

## Files to Modify

### 6. `model/main.py`
- Add `from model.sac.train import train_sac, evaluate_sac`
- Add `run_sac_training()` and `run_sac_eval()` functions (following DQN pattern)
- Add `"sac"` and `"sac-eval"` to argparse choices + dispatch

### 7. `Makefile`
- Add `sac` and `sac-eval` targets (following dqn/dqn-eval pattern)
- Add to `.PHONY` and `help`

## Wrapper Stacking Order

```
Raw MineRL env
  → ContinuousActionWrapper  (5D float → MineRL dict, reward shaping using location_stats)
    → GrayscaleWrapper        (extract pov, RGB→gray, resize 64×64)
      → FrameStackWrapper     (stack 4 frames → 4×64×64 uint8)
        → SAC Agent
```

ContinuousActionWrapper must be innermost because it needs `obs['location_stats']` for reward shaping, which GrayscaleWrapper strips away.

## Implementation Order

1. Create `model/sac/__init__.py`, `networks.py`, `replay_buffer.py`
2. Create `model/sac/wrappers.py` (ContinuousActionWrapper + copied observation wrappers)
3. Create `model/sac/train.py` (SACAgent, train_sac, evaluate_sac)
4. Modify `model/main.py` (add sac/sac-eval modes)
5. Modify `Makefile` (add targets)

## Verification

1. Run `make sac` for a short training session — verify no shape errors, losses decrease
2. Monitor: critic_loss, actor_loss, alpha value, episode rewards
3. Run `make sac-eval` to visualize the trained agent with GUI
