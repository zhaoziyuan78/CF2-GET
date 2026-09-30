import pytest
import torch
from omegaconf import OmegaConf

from cfmusic.codec.bar_chunk_vae import BarChunkVAE
from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.codec.warm_start import initialize_longer_codec
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig
from cfmusic.tokenization.beat_vocabulary import BeatVocabulary


def model(chunk_count=2):
    return BarChunkVAE(
        vocab_size=593,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=2,
        latent_tokens=4 * chunk_count,
        latent_dim=8,
        max_sequence_length=256,
        chunk_bars=2,
        total_bars=2 * chunk_count,
        chunk_max_sequence_length=128,
        pad_id=429,
        bos_id=428,
        eos_id=427,
        unk_id=453,
        vocabulary=BeatVocabulary(),
        dropout=0,
    )


def tokens():
    # Include multitrack/drum events and an unchanged sustain across the boundary.
    body = [426, 425, 297, 249, 54, 161, 504, 541, 27, 181, 503, 503, 503]
    return torch.tensor([[428, 434, 443, *body, *body, *body, *body, 427]])


def test_split_is_lossless_at_internal_sustains_and_aligns_teacher_forcing():
    torch.set_num_threads(1)
    m = model().eval()
    x = tokens()
    chunks, lengths = m.split_tokens(x)
    assert torch.equal(torch.cat([chunks[0, : lengths[0] - 1], chunks[1, 3 : lengths[1]]]), x[0])
    z = m.encode_mean(x, x.ne(m.pad_id))
    assert z.shape == (1, 8, 8)
    raw = TransformerVAE.decode_teacher_forced(m, chunks[:, :-1], z.reshape(2, 4, 8))
    expected = torch.cat([raw[0, : lengths[0] - 2], raw[1, 2 : lengths[1] - 1]])
    actual = m.decode_teacher_forced(x[:, :-1], z)
    torch.testing.assert_close(actual[0], expected)
    out, posterior = m(x, x.ne(m.pad_id))
    torch.testing.assert_close(out, actual)
    torch.testing.assert_close(posterior.mean, z)


def test_training_updates_encoder_decoder_with_unmasked_boundaries():
    m = model().train()
    m.decoder_token_dropout = 0.95
    x = tokens().repeat(2, 1)
    logits, posterior = m(x, x.ne(m.pad_id))
    assert logits.shape == (2, x.shape[1] - 1, 593)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten())
    (loss + posterior.mean.square().mean() * 0.0001).backward()
    assert m.encoder.layers[0].linear1.weight.grad is not None
    assert m.decoder.layers[0].linear1.weight.grad is not None


def test_generation_needs_only_latent_and_does_not_fake_completion(monkeypatch):
    m = model().eval()
    bodies = [[428, 434, 443, *([426, 503, 503, 503, 503] * 2), 427]] * 2

    def generate(_self, latent, **kwargs):
        assert latent.shape == (2, 4, 8)
        return torch.tensor(bodies, device=latent.device)

    monkeypatch.setattr(TransformerVAE, "generate", generate)
    z = torch.randn(1, 8, 8)
    result = m.generate(z, min_bars=4, max_bars=4, max_length=256)
    assert result[0].eq(426).sum() == 4 and result[0, -1] == 427
    # A partial first chunk cannot become a 'complete' full clip by concatenation.
    bodies = [[*bodies[0][:-1], 429], bodies[1]]
    assert 427 not in m.generate(z)[0]


@pytest.mark.parametrize("chunk_count", [2, 3, 4])
@pytest.mark.parametrize("budget", [96, 97, 98, 99])
@pytest.mark.parametrize("termination", ["none", "mixed", "all"])
def test_chunk_budget_exhaustion_never_overflows_or_fakes_eos(
    monkeypatch, chunk_count, budget, termination,
):
    m = model(chunk_count).eval()
    generated = []

    def generate(_self, latent, *, max_length, **kwargs):
        assert max_length <= m.chunk_max_sequence_length
        for chunk in range(chunk_count):
            values = [m.bos_id, 434, 443, *([503] * (max_length - 3))]
            if termination == "all" or (termination == "mixed" and chunk == 0):
                values[-1] = m.eos_id
            generated.append(values)
        return torch.tensor(generated, device=latent.device)

    monkeypatch.setattr(TransformerVAE, "generate", generate)
    result = m.generate(torch.randn(1, 4 * chunk_count, 8), max_length=budget)[0]
    expected = []
    for chunk, values in enumerate(generated):
        payload = values[:-1] if values[-1] == m.eos_id else values
        expected.extend(payload if chunk == 0 else payload[3:])
    if termination == "all":
        expected.append(m.eos_id)
    assert result.tolist() == expected  # No truncation or synthetic completion.
    assert len(result) <= budget
    assert bool(result.eq(m.eos_id).any()) == (termination == "all")


def test_full_chunk_capacity_remains_unchanged(monkeypatch):
    m = model().eval()

    def generate(_self, latent, *, max_length, **kwargs):
        assert max_length == 128  # Both old and corrected 256-token full budgets.
        return torch.tensor([[m.bos_id, 434, 443, *([503] * 125)]] * 2)

    monkeypatch.setattr(TransformerVAE, "generate", generate)
    result = m.generate(torch.randn(1, 8, 8), max_length=256)
    assert result.shape == (1, 253)
    assert not result.eq(m.eos_id).any()


def test_factory_checkpoint_roundtrip_and_old_weight_initialization():
    m = model()
    old = TransformerVAE(
        vocab_size=593,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=2,
        latent_tokens=4,
        latent_dim=8,
        max_sequence_length=64,
        dropout=0,
    )
    initialize_longer_codec(m, old.state_dict())
    torch.testing.assert_close(m.latent_pool.queries, old.latent_pool.queries)
    cfg = OmegaConf.create(
        dict(
            type="bar_chunk_vae",
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=2,
            ff_multiplier=4,
            dropout=0,
            latent_tokens=8,
            latent_dim=8,
            max_sequence_length=256,
            chunk_bars=2,
            total_bars=4,
            chunk_max_sequence_length=128,
            training={},
        )
    )
    other = codec_from_config(cfg, BeatTokenizer(BeatTokenizerConfig(max_sequence_length=256)))
    other.load_state_dict(m.state_dict())
    torch.testing.assert_close(
        other.encode_mean(tokens(), tokens().ne(429)), m.encode_mean(tokens(), tokens().ne(429))
    )


