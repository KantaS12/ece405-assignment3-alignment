import torch
from sft_helper import masked_normalize

def compute_group_normalized_rewards(
    reward_fn,
    rollout_reponses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group sizes.
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truth,
                   producing a dict with keys "reward", "format_reward", and "answer_reward"
        rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size =
                            n_prompts_per_rollout_batch * group_size
        repeated_ground_truths: list[str] Ground truths for the examples. The length of this list is rollout_batch_size,
                                 because the ground truth for each example is repeated group_size times
        group_size: int Number of responses per question (group)
        advantage_epos: float Small constant to avoid division by zero in normalization
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean
    Returns:
        advantages shape (rollout_batch_size,). Group-normalized reward for each rollout response
        raw_rewards shape (rollout_batch_size,). Unnormalized reward for each rollout response
        metadata your choice of other statistic to log (e.g. mean, std, max/min of rewards)

    """

    reward_dicts = [reward_fn(response, gt)
                    for response, gt in zip(rollout_reponses, repeated_ground_truths)]
    rewards = torch.tensor([rd["reward"] for rd in reward_dicts])
    format_rewards = [rd["format_reward"] for rd in reward_dicts]
    answer_rewards = [rd["answer_reward"] for rd in reward_dicts]

    # Reshape rewards to (num_groups, group_size)
    num_groups = len(rollout_reponses) // group_size
    rewards = rewards.view(num_groups, group_size)

    # Compute mean and std for each group
    group_means = rewards.mean(dim=1, keepdim=True)
    group_stds = rewards.std(dim=1, keepdim=True) + advantage_eps

    if normalize_by_std:
        advantages = (rewards - group_means) / group_stds
    else:
        advantages = rewards - group_means

    # Flatten back to (rollout_batch_size,)
    advantages = advantages.view(-1)
    raw_rewards = rewards.view(-1)

    n = len(rollout_reponses)
    metadata = {
        "mean_reward": rewards.mean().item(),
        "std_reward": rewards.std().item(),
        "max_reward": rewards.max().item(),
        "min_reward": rewards.min().item(),
        "mean_format_reward": sum(format_rewards) / n,
        "mean_answer_reward": sum(answer_rewards) / n,
    }

    return_tuple = (advantages, raw_rewards, metadata)
    return return_tuple
                                 

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward 
    or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
                          each token
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to 
        be aggregated across the batch and sequence dimensions in the training loop).
    """

    # Expand rewards/advantages to match the shape of policy_log_probs
    expanded_rewards = raw_rewards_or_advantages.expand_as(policy_log_probs)

    # Compute the policy gradient loss
    loss = -expanded_rewards * policy_log_probs

    return loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
                          probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
                       from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).

    Returns:
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
             loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS.
    """

    # Compute the probability ratio
    log_prob_diff = policy_log_probs - old_log_probs
    ratio = torch.exp(log_prob_diff)

    # Compute the clipped ratio

    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    # Compute the unclipped and clipped losses
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * clipped_ratio

    # Take the element-wise maximum (most conservative / pessimistic bound)
    loss = torch.max(unclipped_loss, clipped_loss)

    # Metadata for logging
    metadata = {
        "ratio": ratio.mean().item(),
        "clipped_ratio": clipped_ratio.mean().item(),
        "unclipped_loss": unclipped_loss.mean().item(),
        "clipped_loss": clipped_loss.mean().item(),
        "percent_clipped": ((ratio > 1 + cliprange) | (ratio < 1 - cliprange)).float().mean().item()
    }

    return_tuple = (loss, metadata)
    return return_tuple

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those lements where masks == 1.

    Args:

    tensor: torch.Tensor The data to be averaged
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.

    Returns:

    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """

    # Calculate masked tensor using element-wise multiplication
    masked_tensor = mask * tensor

    # Compute thet mean 

    if dim != None:
        masked_sum = masked_tensor.sum(dim=dim)

        masked_mean = masked_sum / mask.sum(dim=dim).float()

    else:
        # If dim is None, we will compute the mean over all masked elements
        masked_mean = torch.mean(masked_tensor[mask == 1])

    return masked_mean



def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
                        policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
                        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange Clip parameter ε for GRPO-Clip.

    Returns:
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                    this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
                    might want to log
    """


    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    # Average over sequence (masked), then over batch -> scalar
    # Masked Mean
    masked_loss = masked_mean(loss, response_mask, dim=1).mean()

    # Masked Normalize
    # masked_loss = masked_normalize(loss, response_mask, dim=1).mean()

    # Scale for gradient accumulation and backprop
    scaled_loss = masked_loss / gradient_accumulation_steps
    scaled_loss.backward()

    return (scaled_loss.detach(), metadata)
        

def grpo_microbatch_train_step_mean_normalized(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
                        policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
                        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange Clip parameter ε for GRPO-Clip.

    Returns:
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                    this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
                    might want to log
    """


    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    # Masked Normalize
    masked_loss = masked_normalize(loss, response_mask, dim=1).mean()

    # Scale for gradient accumulation and backprop
    scaled_loss = masked_loss / gradient_accumulation_steps
    scaled_loss.backward()

    return (scaled_loss.detach(), metadata)