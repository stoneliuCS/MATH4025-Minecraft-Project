import torch

# compute loss following the Bradley-Terry model
def preference_loss(reward_model, seg_a, seg_b, prefs):
    """
    seg_a, seg_b: (batch, T, obs_dim) trajectory segments
    prefs: (batch,) float — 1.0 if seg_a preferred, 0.0 if seg_b preferred, 0.5 for ties
    """
    r_a = reward_model(seg_a)  # (batch,)
    r_b = reward_model(seg_b)  # (batch,)

    # follows the formulas on page 5 of the paper by Cristiano et al. 

    # P(a preferred) = exp(r_a) / (exp(r_a) + exp(r_b))
    logits = torch.stack([r_a, r_b], dim=1)       # (batch, 2)
    log_probs = torch.log_softmax(logits, dim=1)   # log P(a), log P(b)

    # Cross-entropy against human preference labels
    loss = -(prefs * log_probs[:, 0] + (1 - prefs) * log_probs[:, 1])
    return loss.mean()