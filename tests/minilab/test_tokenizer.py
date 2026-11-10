from packages.minilab.tokenizer import (
    IM_END,
    IM_START,
    StreamDecoder,
    Tokenizer,
    merge,
    pre_tokenize,
    train_bpe,
)

TEXT = "le chat et le chien. Le chat dort, le chien aussi : été, déjà, voilà ! 123456"


def test_merge_replaces_pairs_from_left_to_right():
    assert merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge([5, 5, 5], (5, 5), 7) == [7, 5]


def test_pre_tokenization_keeps_the_space_with_the_next_word():
    assert pre_tokenize("le chat, 12345") == ["le", " chat", ",", " 123", "45"]


def test_training_learns_the_requested_number_of_merges():
    merges = train_bpe(TEXT * 20, vocab_size=280)
    assert len(merges) == 280 - 256 - 2
    assert merges == train_bpe(TEXT * 20, vocab_size=280)  # deterministic
    # « ␣c » (␣chat, ␣chien) and « ch » are tied at 4: the smallest pair wins.
    assert merges[0] == (32, 99)


def test_encode_decode_round_trip_and_compression():
    tokenizer = Tokenizer(train_bpe(TEXT * 20, vocab_size=300))
    ids = tokenizer.encode(TEXT)
    assert tokenizer.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8"))
    assert tokenizer.vocab_size == 300
    # Unknown characters still work: they fall back on their bytes.
    assert tokenizer.decode(tokenizer.encode("🙂 ж")) == "🙂 ж"


def test_special_tokens_only_when_allowed():
    tokenizer = Tokenizer(train_bpe(TEXT * 20, vocab_size=300))
    text = f"{IM_START}user\nBonjour{IM_END}"
    ids = tokenizer.encode(text, allow_special=True)
    assert ids[0] == tokenizer.special[IM_START] == 298
    assert ids[-1] == tokenizer.special[IM_END] == 299
    assert tokenizer.special[IM_END] not in tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_stream_decoder_waits_for_the_missing_bytes():
    tokenizer = Tokenizer([])  # bytes only: « é » is two tokens
    stream = StreamDecoder(tokenizer)
    outputs = [stream.feed(token_id) for token_id in tokenizer.encode("é!")]
    assert outputs == ["", "é", "!"]
    assert stream.flush() == ""
