import torch

from packages.minilab.generate import SamplingSettings, generate, next_token_probabilities
from packages.minilab.model import Config, KVCache, MiniLabModel

TINY = Config(vocab_size=30, context_length=24, d_model=16, n_head=2, n_layer=2)


def test_greedy_puts_everything_on_the_favourite():
    probabilities = next_token_probabilities(torch.tensor([1.0, 3.0, 2.0]), SamplingSettings())
    assert probabilities.tolist() == [0.0, 1.0, 0.0]


def test_temperature_top_k_and_top_p():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    plain = next_token_probabilities(logits, SamplingSettings(temperature=1.0))
    assert torch.allclose(plain, torch.tensor([0.5, 0.3, 0.15, 0.05]))
    cold = next_token_probabilities(logits, SamplingSettings(temperature=0.5))
    assert cold[0] > 0.5  # a low temperature sharpens the distribution
    top_k = next_token_probabilities(logits, SamplingSettings(temperature=1.0, top_k=2))
    assert torch.allclose(top_k, torch.tensor([0.625, 0.375, 0.0, 0.0]))
    top_p = next_token_probabilities(logits, SamplingSettings(temperature=1.0, top_p=0.8))
    assert torch.allclose(top_p, torch.tensor([0.625, 0.375, 0.0, 0.0]))
    wider = next_token_probabilities(logits, SamplingSettings(temperature=1.0, top_p=0.81))
    assert torch.allclose(wider, torch.tensor([0.5, 0.3, 0.15, 0.0]) / 0.95)


def test_kv_cache_gives_the_same_tokens_and_logits():
    torch.manual_seed(0)
    model = MiniLabModel(TINY)
    prompt = [1, 2, 3, 4]
    settings = SamplingSettings(temperature=0.9, top_k=10, seed=3)
    cache = KVCache(TINY.n_layer)
    with_cache = list(generate(model, prompt, 15, settings, cache=cache))
    without = list(generate(model, prompt, 15, settings, use_cache=False))
    assert [s.token_id for s in with_cache] == [s.token_id for s in without]
    for a, b in zip(with_cache, without, strict=True):
        assert torch.allclose(a.logits, b.logits, atol=1e-5)
    # Prefill of the 4 prompt tokens, then one token per decode step, in every layer.
    assert cache.projected == TINY.n_layer * (4 + 14)
    assert cache.length == 4 + 14


def test_generation_stops_on_the_stop_token_and_the_window():
    torch.manual_seed(0)
    model = MiniLabModel(TINY)
    first = next(generate(model, [1, 2], 5)).token_id
    steps = list(generate(model, [1, 2], 10, stop_id=first))
    assert [s.token_id for s in steps] == [first]
    assert len(list(generate(model, [1] * 20, 50))) == TINY.context_length - 20
