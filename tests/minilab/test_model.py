import math

import torch

from packages.minilab.model import Config, MiniLabModel, count_parameters, load, save

TINY = Config(vocab_size=50, context_length=16, d_model=32, n_head=4, n_layer=2)


def test_shapes_and_parameter_count():
    torch.manual_seed(0)
    model = MiniLabModel(TINY)
    logits, loss = model(torch.randint(0, 50, (3, 10)))
    assert logits.shape == (3, 10, 50) and loss is None
    d = TINY.d_model
    block = 4 * d * d + (d * 4 * d + 4 * d) + (4 * d * d + d) + 2 * (2 * d)
    expected = 50 * d + 16 * d + TINY.n_layer * block + 2 * d  # the head shares the embeddings
    assert count_parameters(model) == expected


def test_causal_mask_hides_the_future():
    torch.manual_seed(0)
    model = MiniLabModel(TINY).eval()
    ids = torch.randint(0, 50, (1, 8))
    changed = ids.clone()
    changed[0, 5:] = (changed[0, 5:] + 1) % 50
    with torch.no_grad():
        before, after = model(ids)[0], model(changed)[0]
    assert torch.allclose(before[0, :5], after[0, :5])  # past positions do not move
    assert not torch.allclose(before[0, 5:], after[0, 5:])


def test_initial_loss_is_close_to_ln_of_the_vocabulary():
    torch.manual_seed(0)
    model = MiniLabModel(TINY)
    ids = torch.randint(0, 50, (8, 17))
    _, loss = model(ids[:, :-1], ids[:, 1:])
    assert abs(loss.item() - math.log(50)) < 0.1


def test_ignored_targets_do_not_count_in_the_loss():
    torch.manual_seed(0)
    model = MiniLabModel(TINY)
    ids = torch.randint(0, 50, (1, 9))
    targets = ids[:, 1:].clone()
    masked = targets.clone()
    masked[0, :4] = -100
    logits, _ = model(ids[:, :-1])
    expected = torch.nn.functional.cross_entropy(logits[0, 4:], targets[0, 4:])
    assert torch.isclose(model(ids[:, :-1], masked)[1], expected)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = MiniLabModel(TINY).eval()
    save(model, tmp_path / "model.pt")
    ids = torch.randint(0, 50, (1, 6))
    assert torch.equal(model(ids)[0], load(tmp_path / "model.pt")(ids)[0])
