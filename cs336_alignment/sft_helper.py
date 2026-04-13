import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):

    # Tokenize prompts and outputs separately, then concatenate
    all_prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    all_output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    # Concatenate prompt and output token ids for each example
    all_full_ids = [p + o for p, o in zip(all_prompt_ids, all_output_ids)]

    # Pad sequences to the same length
    max_len = max(len(f) for f in all_full_ids)
    pad_id = tokenizer.eos_token_id

    # Pad with pad_id (e.g., EOS token) to the right
    padded = [f + [pad_id] * (max_len - len(f)) for f in all_full_ids]
    full_tensor = torch.tensor(padded)  # (batch, max_len)

    input_ids = full_tensor[:, :-1]
    labels = full_tensor[:, 1:]

    # Create response mask that is True for positions corresponding to the output (response) tokens, and False for prompt tokens
    seq_len = max_len - 1
    response_mask = torch.zeros(len(prompt_strs), seq_len, dtype=torch.bool)
    for i, (p_ids, o_ids) in enumerate(zip(all_prompt_ids, all_output_ids)):
        start = len(p_ids) - 1
        end = start + len(o_ids)
        response_mask[i, start:end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


from scipy.special import logsumexp
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: (batch_size, seq_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)  # (batch_size, seq_len)
    return entropy


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:

    # input_ids, labels: (batch_size, seq_len)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Gather log probabilities of the true next tokens
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        labels_flat = labels.view(-1)  # (batch_size * seq_len)
        log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)  # (batch_size * seq_len, vocab_size)
        token_log_probs_flat = log_probs_flat[torch.arange(batch_size * seq_len), labels_flat]  # (batch_size * seq_len)
        token_log_probs = token_log_probs_flat.view(batch_size, seq_len)  # (batch_size, seq_len)

        result = {"log_probs": token_log_probs}

        if return_token_entropy:
            entropy = compute_entropy(logits)  # (batch_size, seq_len)
            result["token_entropy"] = entropy

        return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:

    masked_sum = (tensor * mask).sum(dim=dim)
    return masked_sum / normalize_constant



def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    masked_log_probs = policy_log_probs * response_mask
    per_sample_loss = -masked_log_probs.sum(dim=-1) / (normalize_constant * gradient_accumulation_steps)
    loss = per_sample_loss.mean()

    loss.backward()

    metrics = {
        "loss": loss.item(),
        "avg_token_log_prob": masked_log_probs.sum().item() / response_mask.sum().item(),
    }

    return loss, metrics


def log_generations(
    input_prompts: list[str],
    rollout_responses: list[str],
    ground_truths: list[str],
    reward_dicts: list[dict],
    token_entropies: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
    log_path: str | None = None,
):
    """Log per-sample generation info and aggregate batch statistics.

    Args:
        input_prompts: list of prompt strings (length = rollout_batch_size).
        rollout_responses: list of generated response strings.
        ground_truths: list of ground-truth answer strings.
        reward_dicts: list of dicts with keys "reward", "format_reward", "answer_reward".
        token_entropies: optional tensor of shape (batch, seq_len) with per-token entropy.
        response_mask: optional bool tensor of shape (batch, seq_len) masking response tokens.
        log_path: if given, append log output to this file; otherwise print to stdout.
    """
    import io
    buf = io.StringIO()

    def out(s=""):
        buf.write(s + "\n")

    batch_size = len(input_prompts)

    # Per-sample logging
    for i in range(batch_size):
        out(f"{'='*60}")
        out(f"[Sample {i+1}/{batch_size}]")
        out(f"  PROMPT      : {input_prompts[i]}")
        out(f"  RESPONSE    : {rollout_responses[i]}")
        out(f"  GROUND TRUTH: {ground_truths[i]}")
        rd = reward_dicts[i]
        out(f"  REWARD      : total={rd.get('reward', float('nan')):.4f} | "
            f"format={rd.get('format_reward', float('nan')):.4f} | "
            f"answer={rd.get('answer_reward', float('nan')):.4f}")

        # Per-sample token entropy (masked mean over response tokens)
        if token_entropies is not None and response_mask is not None:
            mask_i = response_mask[i].bool()
            ent_i = token_entropies[i]
            if mask_i.any():
                avg_ent = ent_i[mask_i].mean().item()
            else:
                avg_ent = float('nan')
            out(f"  TOKEN ENTROPY (avg): {avg_ent:.4f}")

        # Per-sample response length (number of response tokens)
        if response_mask is not None:
            resp_len = response_mask[i].sum().item()
        else:
            resp_len = len(rollout_responses[i].split())
        out(f"  RESPONSE LEN: {resp_len}")

    out(f"{'='*60}")
    out("[Batch Statistics]")

    # Average token entropy across the batch
    if token_entropies is not None and response_mask is not None:
        mask_all = response_mask.bool()
        # weighted mean: sum of entropy over response tokens / total response tokens
        total_response_tokens = mask_all.sum().item()
        if total_response_tokens > 0:
            avg_entropy = (token_entropies * mask_all).sum().item() / total_response_tokens
        else:
            avg_entropy = float('nan')
        out(f"  Avg token entropy       : {avg_entropy:.4f}")

    # Response lengths
    if response_mask is not None:
        lengths = response_mask.bool().sum(dim=-1).float()  # (batch,)
    else:
        lengths = torch.tensor([len(r.split()) for r in rollout_responses], dtype=torch.float)

    avg_len = lengths.mean().item()

    # Determine correctness: answer_reward > 0
    is_correct = torch.tensor(
        [rd.get('answer_reward', 0.0) > 0.0 for rd in reward_dicts], dtype=torch.bool
    )
    correct_lengths = lengths[is_correct]
    incorrect_lengths = lengths[~is_correct]

    avg_len_correct = correct_lengths.mean().item() if correct_lengths.numel() > 0 else float('nan')
    avg_len_incorrect = incorrect_lengths.mean().item() if incorrect_lengths.numel() > 0 else float('nan')

    out(f"  Avg response length     : {avg_len:.1f} tokens")
    out(f"  Avg len (correct)       : {avg_len_correct:.1f} tokens  (n={is_correct.sum().item()})")
    out(f"  Avg len (incorrect)     : {avg_len_incorrect:.1f} tokens  (n={(~is_correct).sum().item()})")
    out(f"{'='*60}")

    log_text = buf.getvalue()
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(log_text)
    else:
        print(log_text, end="")

