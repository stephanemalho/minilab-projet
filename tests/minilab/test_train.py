import torch

from packages.minilab.model import Config, MiniLabModel
from packages.minilab.train import TrainingSettings, get_batch, learning_rate, split, train

TINY = Config(vocab_size=20, context_length=8, d_model=16, n_head=2, n_layer=1)


def test_split_keeps_ninety_percent_for_training():
    train_data, val_data = split(list(range(100)))
    assert len(train_data) == 90 and len(val_data) == 10
    assert train_data[-1].item() == 89 and val_data[0].item() == 90


def test_targets_are_the_inputs_shifted_by_one():
    data = torch.arange(50)
    inputs, targets = get_batch(data, 4, 8, torch.Generator().manual_seed(0))
    assert inputs.shape == targets.shape == (4, 8)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert torch.equal(targets[:, -1], inputs[:, -1] + 1)


def test_learning_rate_warms_up_then_decays():
    settings = TrainingSettings(steps=100, learning_rate=1.0, warmup=10)
    rates = [learning_rate(step, settings) for step in range(100)]
    assert rates[0] == 0.1 and rates[9] == 1.0
    assert rates[10] > rates[50] > rates[99] >= 0.1


def test_training_lowers_the_loss_and_keeps_the_best_weights():
    torch.manual_seed(0)
    pattern = torch.tensor([1, 2, 3, 4, 5] * 40)
    model = MiniLabModel(TINY)
    settings = TrainingSettings(
        steps=60, batch_size=8, learning_rate=1e-2, warmup=5, eval_every=20, eval_batches=2
    )
    history = train(model, pattern, pattern, settings)
    first, last = history.points[0], history.points[-1]
    assert last[1] < first[1] / 3  # a repeating pattern is easy to learn
    assert history.best_val == min(point[2] for point in history.points)
    assert history.best_state is not None
