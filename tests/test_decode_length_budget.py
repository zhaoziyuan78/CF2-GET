"""Output budgets must not inherit a shorter input-tokenizer limit."""

import pytest

from cfmusic.codec.decoder_adaptation import decode_completion
from cfmusic.commands.generate_counterfactuals import _decode_length_limit
from cfmusic.tokenization.beat_vocabulary import BeatVocabulary


@pytest.mark.parametrize("count", [None, 100, 4000, float("nan")])
def test_full_output_capacity_ignores_input_and_source_caps(count):
    assert _decode_length_limit(
        tokenizer_limit=5120, codec_limit=10240, full_decode_budget=True,
        source_token_count=count,
    ) == 10240


@pytest.mark.parametrize("count,expected", [(None, 5120), (1000, 1100), (8000, 5120),
                                            (1, 16), (float("nan"), 5120)])
def test_legacy_adaptive_budget_is_unchanged(count, expected):
    assert _decode_length_limit(
        tokenizer_limit=5120, codec_limit=10240, full_decode_budget=False,
        source_token_count=count,
    ) == expected


def test_decoder_capacity_is_respected():
    assert _decode_length_limit(
        tokenizer_limit=5120, codec_limit=2560, full_decode_budget=False,
    ) == 2560
    with pytest.raises(ValueError, match="positive"):
        _decode_length_limit(tokenizer_limit=5120, codec_limit=0, full_decode_budget=True)


def test_incomplete_chunk_is_budget_exhausted_even_below_joint_cap():
    vocabulary = BeatVocabulary()
    tokens = [vocabulary.bos_id, vocabulary.bar_id, vocabulary.id("REST")]
    result = decode_completion(tokens, vocabulary, requested_bars=16, token_budget=10240,
                               chunked=True)
    assert not result["complete"] and result["budget_exhausted"]
    legacy = decode_completion(tokens, vocabulary, requested_bars=16, token_budget=10240)
    assert not legacy["budget_exhausted"]
