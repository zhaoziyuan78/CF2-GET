"""Shared-weight VAE over bar-aligned chunks, with one joint clip latent.

The Gaussian posterior and decoder factorize over chunks. Downstream transport
sees the concatenation, not separately sampled clips. Generation has no access
to input tokens, original MIDI, or an encoder-side cache.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cfmusic.codec.base import DiagonalGaussian
from cfmusic.codec.transformer_vae import TransformerVAE


class BarChunkVAE(TransformerVAE):
    def __init__(self, *, chunk_bars=8, total_bars=16, chunk_max_sequence_length=5120, **kwargs):
        total_latents = int(kwargs.pop("latent_tokens"))
        full_limit = int(kwargs.pop("max_sequence_length"))
        if total_bars % chunk_bars or total_bars <= chunk_bars:
            raise ValueError("Full clip must contain an integer number of chunks > 1")
        self.chunk_count = total_bars // chunk_bars
        if total_latents % self.chunk_count:
            raise ValueError("Latent token count must be divisible by chunk count")
        super().__init__(
            latent_tokens=total_latents // self.chunk_count,
            max_sequence_length=chunk_max_sequence_length,
            **kwargs,
        )
        if self.grammar is None or not self.grammar.beat_scheme:
            raise ValueError("BarChunkVAE requires the BEAT vocabulary")
        self.chunk_latent_tokens = self.latent_tokens
        self.latent_tokens = total_latents
        self.max_sequence_length = full_limit
        self.chunk_max_sequence_length = chunk_max_sequence_length
        self.chunk_bars, self.total_bars = chunk_bars, total_bars

    def _embed(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] > self.chunk_max_sequence_length:
            raise ValueError("Chunk exceeds positional capacity; truncation is forbidden")
        return self.token_embedding(tokens) + self.position_embedding[: tokens.shape[1]]

    def split_tokens(self, tokens: Tensor) -> tuple[Tensor, list[int]]:
        """Split a complete clip or its teacher-forcing prefix without altering patterns.

        In particular, sustain patterns across the internal boundary are retained,
        not converted into note onsets. Metadata are duplicated only for the local
        network inputs; generated metadata come exclusively from the decoder.
        """
        vocabulary = self.grammar.vocabulary
        chunks, lengths = [], []
        for values in tokens.detach().cpu().tolist():
            values = [v for v in values if v != self.pad_id]
            if self.eos_id in values:
                values = values[: values.index(self.eos_id)]
            bars = [i for i, value in enumerate(values) if value == vocabulary.bar_id]
            if len(bars) != self.total_bars or bars[0] != 3:
                raise ValueError(f"Expected a complete {self.total_bars}-bar BEAT input")
            if values[0] != self.bos_id:
                raise ValueError("Missing BEAT header")
            for chunk in range(self.chunk_count):
                start = bars[chunk * self.chunk_bars]
                end = (
                    bars[(chunk + 1) * self.chunk_bars]
                    if chunk + 1 < self.chunk_count
                    else len(values)
                )
                sequence = [*values[:3], *values[start:end], self.eos_id]
                if len(sequence) > self.chunk_max_sequence_length:
                    raise ValueError("Chunk exceeds positional capacity; truncation is forbidden")
                chunks.append(sequence)
                lengths.append(len(sequence))
        packed = tokens.new_full((len(chunks), max(lengths)), self.pad_id)
        for row, sequence in enumerate(chunks):
            packed[row, : len(sequence)] = torch.tensor(sequence, device=tokens.device)
        return packed, lengths

    def encode_distribution(self, tokens: Tensor, attention_mask: Tensor) -> DiagonalGaussian:
        chunks, _ = self.split_tokens(tokens.masked_fill(~attention_mask.bool(), self.pad_id))
        posterior = super().encode_distribution(chunks, chunks.ne(self.pad_id))
        shape = (len(tokens), self.latent_tokens, self.latent_dim)
        return DiagonalGaussian(posterior.mean.reshape(shape), posterior.logvar.reshape(shape))

    def _decode_chunks(self, chunks, lengths, latent, output_length, *, corrupt=False):
        if latent.shape[1:] != (self.latent_tokens, self.latent_dim):
            raise ValueError("Joint latent shape does not match chunk codec")
        prefix = chunks[:, :-1]
        if corrupt and self.decoder_token_dropout:
            eligible = prefix.ne(self.pad_id) & prefix.ne(self.bos_id) & prefix.ne(self.eos_id)
            dropped = torch.rand(prefix.shape, device=prefix.device).lt(self.decoder_token_dropout)
            prefix = prefix.masked_fill(eligible & dropped, self.unk_id)
        logits = super().decode_teacher_forced(
            prefix, latent.reshape(-1, self.chunk_latent_tokens, self.latent_dim)
        )
        joined = []
        for row in range(len(latent)):
            parts = []
            for chunk in range(self.chunk_count):
                index = row * self.chunk_count + chunk
                # Interior EOS and repeated header predictions are not positions
                # in the full-clip target. No synthetic logits replace music events.
                begin = 0 if chunk == 0 else 2
                end = lengths[index] - (1 if chunk + 1 == self.chunk_count else 2)
                parts.append(logits[index, begin:end])
            value = torch.cat(parts)
            if len(value) > output_length:
                raise ValueError("Teacher-forcing alignment overflow")
            joined.append(F.pad(value, (0, 0, 0, output_length - len(value))))
        return torch.stack(joined)

    def decode_teacher_forced(self, tokens: Tensor, latent: Tensor) -> Tensor:
        chunks, lengths = self.split_tokens(tokens)
        return self._decode_chunks(chunks, lengths, latent, tokens.shape[1])

    def forward(self, tokens: Tensor, attention_mask: Tensor):
        chunks, lengths = self.split_tokens(tokens.masked_fill(~attention_mask.bool(), self.pad_id))
        local = super().encode_distribution(chunks, chunks.ne(self.pad_id))
        shape = (len(tokens), self.latent_tokens, self.latent_dim)
        posterior = DiagonalGaussian(local.mean.reshape(shape), local.logvar.reshape(shape))
        latent = posterior.sample() if self.training else posterior.mean
        return self._decode_chunks(
            chunks, lengths, latent, tokens.shape[1] - 1, corrupt=self.training
        ), posterior

    @torch.no_grad()
    def generate(
        self,
        latent: Tensor,
        *,
        max_length=None,
        min_bars=None,
        max_bars=None,
        use_cache=True,
        **kwargs,
    ) -> Tensor:
        if any(b is not None and b != self.total_bars for b in (min_bars, max_bars)):
            raise ValueError("Chunk codec generates the configured full clip length only")
        if not use_cache:
            raise ValueError("Chunk codec generation requires incremental decoding")
        if latent.shape[1:] != (self.latent_tokens, self.latent_dim):
            raise ValueError("Joint latent shape does not match chunk codec")
        budget = self.max_sequence_length if max_length is None else int(max_length)
        # Only the repeated three-token header is guaranteed to disappear at
        # each seam. A budget-exhausted chunk has no EOS to remove, so counting
        # an extra EOS saving can overflow the full budget on incomplete clips.
        # Reserve a fair fixed budget, never source-dependent, for every chunk.
        local_budget = min(
            self.chunk_max_sequence_length,
            (budget + 3 * (self.chunk_count - 1)) // self.chunk_count,
        )
        if local_budget < 4 + self.chunk_bars * 5:
            raise ValueError("Token budget cannot represent complete chunks")
        local = (
            super()
            .generate(
                latent.reshape(-1, self.chunk_latent_tokens, self.latent_dim),
                max_length=local_budget,
                min_bars=self.chunk_bars,
                max_bars=self.chunk_bars,
                use_cache=True,
                **kwargs,
            )
            .cpu()
            .tolist()
        )
        sequences = []
        for row in range(len(latent)):
            sequence, all_finished = [], True
            for chunk in range(self.chunk_count):
                values = [v for v in local[row * self.chunk_count + chunk] if v != self.pad_id]
                finished = self.eos_id in values
                all_finished &= finished
                if finished:
                    values = values[: values.index(self.eos_id)]
                sequence.extend(values if chunk == 0 else values[3:])
            # Do not manufacture a complete sequence when any local decoder hit
            # its token limit. There is no silent padding with empty bars.
            if all_finished:
                sequence.append(self.eos_id)
            if len(sequence) > budget:
                raise RuntimeError("Combined output exceeds declared token budget")
            sequences.append(sequence)
        result = latent.new_full(
            (len(sequences), max(map(len, sequences))), self.pad_id, dtype=torch.long
        )
        for index, sequence in enumerate(sequences):
            result[index, : len(sequence)] = torch.tensor(sequence, device=latent.device)
        return result
