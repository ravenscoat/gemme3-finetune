"""Explicit learning-rate policy for extending the completed short pilot."""


def continuation_scale(step, start, end, warmup=50):
    elapsed = max(0, step - start)
    if elapsed < warmup:
        return elapsed / warmup
    return max(0., (end - step) / max(1, end - start - warmup))
