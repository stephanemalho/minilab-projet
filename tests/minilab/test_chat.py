import pytest
import torch

from packages.minilab.chat import (
    IGNORE,
    SFTSettings,
    answer,
    fine_tune,
    fit_to_window,
    load_dialogues,
    make_examples,
    prompt_ids,
    render,
)
from packages.minilab.model import Config, MiniLabModel
from packages.minilab.tokenizer import IM_END, IM_START, Tokenizer, train_bpe

TOKENIZER = Tokenizer(train_bpe("le chat dort. qui est le chat ? un chat. " * 30, vocab_size=300))
START, END = TOKENIZER.special[IM_START], TOKENIZER.special[IM_END]


def test_template_and_loss_mask():
    messages = [{"role": "user", "content": "qui ?"}, {"role": "assistant", "content": "le chat."}]
    ids, trained = render(messages, TOKENIZER)
    expected = f"{IM_START}user\nqui ?{IM_END}\n{IM_START}assistant\nle chat.{IM_END}\n"
    assert TOKENIZER.decode(ids) == expected
    # Only the answer and its <|im_end|> count in the loss.
    assert (
        TOKENIZER.decode([t for t, keep in zip(ids, trained, strict=True) if keep])
        == f"le chat.{IM_END}"
    )
    assert prompt_ids(messages[:1], TOKENIZER)[-len(TOKENIZER.encode("assistant\n")) - 1] == START


def test_examples_are_shifted_and_masked():
    dialogue = [{"role": "user", "content": "qui ?"}, {"role": "assistant", "content": "le chat."}]
    [(inputs, targets)] = make_examples([dialogue], TOKENIZER, 64)
    ids, trained = render(dialogue, TOKENIZER)
    assert inputs == ids[:-1] and len(targets) == len(inputs)
    assert targets[-2] == END and targets[-1] == IGNORE  # the final line break is not trained
    assert all(t == IGNORE for t, keep in zip(targets, trained[1:], strict=True) if not keep)


def test_window_drops_whole_turns_and_keeps_the_system_prompt():
    history = [
        {"role": "user", "content": "qui est le chat ?"},
        {"role": "assistant", "content": "un chat."},
    ] * 3
    history.append({"role": "user", "content": "le chat dort ?"})
    # 120 tokens minus 16 reserved: the system prompt and the last 3 messages fit (84 tokens),
    # the last 5 would not (114). The oldest turns go, a question with its answer.
    kept, ids = fit_to_window(history, TOKENIZER, 120, 16)
    assert kept[0]["role"] == "system" and kept[1:] == history[-3:]
    assert len(ids) == 84 and len(ids) + 16 <= 120
    with pytest.raises(ValueError):
        fit_to_window(history[-1:], TOKENIZER, 20, 16)


def test_the_project_dialogues_fit_in_the_window():
    from packages.minilab.tokenizer import get_tokenizer

    dialogues = load_dialogues()
    assert len(dialogues) >= 60
    assert all(len(render(d, get_tokenizer())[0]) <= 129 for d in dialogues)


def test_fine_tuning_teaches_the_answer_and_the_stop():
    torch.manual_seed(0)
    model = MiniLabModel(Config(vocab_size=300, context_length=48, d_model=32, n_head=2, n_layer=2))
    dialogue = [{"role": "user", "content": "qui ?"}, {"role": "assistant", "content": "le chat."}]
    examples = make_examples(
        [[{"role": "system", "content": "un chat."}, *dialogue]], TOKENIZER, 48
    )
    losses = fine_tune(model, examples, SFTSettings(steps=60, batch_size=4, learning_rate=1e-2))
    assert losses[-1] < losses[0] / 10
    text, stopped = answer(
        model, TOKENIZER, [{"role": "system", "content": "un chat."}, dialogue[0]], 10
    )
    assert (text, stopped) == ("le chat.", True)
