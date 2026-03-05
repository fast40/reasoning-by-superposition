# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.cache_utils import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        layer_skip=False,
        skip_layer_norm=True,
        skip_inject_layer=1,
        skip_extract_layer=1,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.layer_skip = layer_skip
        self.skip_layer_norm = skip_layer_norm
        self.skip_inject_layer = skip_inject_layer   # num bottom layers to skip (KV-only, output discarded)
        self.skip_extract_layer = skip_extract_layer  # num top layers to skip (feedback extracted before these)

        # NOTE: we access layers/ln_f/lm_head via properties (not stored as
        # self.xxx attributes) to avoid registering them as duplicate submodules,
        # which would break FSDP's auto-wrap.
        self.embedding = self.base_causallm.get_input_embeddings()

    @property
    def layers(self):
        return self.base_causallm.model.layers

    @property
    def ln_f(self):
        return self.base_causallm.model.norm

    @property
    def lm_head(self):
        return self.base_causallm.lm_head

    def _forward_latent_pass(
        self, original_embeds, continuous_thought,
        attention_mask, position_ids, past_key_values,
    ):
        """
        Manual forward for latent recurrence with configurable layer skip.

        Layout (N total layers, I = skip_inject_layer, E = skip_extract_layer):
          Bottom  layers [0, I)   : run on original_embeds (KV cache only, output discarded)
          Middle  layers [I, N-E) : run on continuous_thought (the recurrence loop)
          Top     layers [N-E, N) : run on middle output (KV cache + produces logits)

        The new continuous thought is extracted at the boundary between middle
        and top layers, optionally with ln_f applied (controlled by skip_layer_norm).

        Returns: (logits, new_continuous_thought, full_kv_cache)
        """
        n = len(self.layers)
        I = self.skip_inject_layer
        E = self.skip_extract_layer
        device = continuous_thought.device

        cache = DynamicCache()
        if past_key_values:
            for layer_idx, (k, v) in enumerate(past_key_values):
                cache.update(k, v, layer_idx)

        past_seen_tokens = cache.get_seq_length() if len(cache) > 0 else 0
        seq_len = continuous_thought.shape[1]

        position_embeddings = self.base_causallm.model.rotary_emb(
            continuous_thought, position_ids,
        )

        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=device,
        )
        causal_mask = self.base_causallm.model._update_causal_mask(
            attention_mask, continuous_thought, cache_position,
            past_key_values=cache, output_attentions=False,
        )

        # --- Bottom layers [0, I): original_embeds for KV cache, output discarded ---
        h_bottom = original_embeds
        for i in range(I):
            out_i = self.layers[i](
                h_bottom,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            h_bottom = out_i[0]

        # --- Middle layers [I, N-E): continuous thought recurrence ---
        h = continuous_thought
        for i in range(I, n - E):
            out_i = self.layers[i](
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            h = out_i[0]

        new_continuous_thought = self.ln_f(h) if self.skip_layer_norm else h

        # --- Top layers [N-E, N): KV cache + logits ---
        for i in range(n - E, n):
            out_i = self.layers[i](
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            h = out_i[0]

        logits = self.lm_head(self.ln_f(h))

        kv_tuple = tuple(
            (cache.key_cache[i], cache.value_cache[i]) for i in range(n)
        )

        return logits, new_continuous_thought, kv_tuple

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        # Whether we actually need the manual layer-by-layer forward
        use_manual_forward = (
            self.layer_skip
            and (self.skip_inject_layer > 0 or self.skip_extract_layer > 0)
        )

        if use_manual_forward:
            original_inputs_embeds = inputs_embeds.clone()

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass: full model on pre-latent tokens
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

                logits.append(outputs.logits)

                if use_manual_forward:
                    h = outputs.hidden_states[-(1 + self.skip_extract_layer)]
                    hidden_states = self.ln_f(h) if self.skip_layer_norm else h
                else:
                    hidden_states = outputs.hidden_states[-1]
                kv_cache = outputs.past_key_values

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                if use_manual_forward:
                    ct = inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ]
                    orig = original_inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ]

                    out_logits, hidden_states, kv_cache = self._forward_latent_pass(
                        orig, ct,
                        attention_mask[:, : next_compute_range[1]],
                        position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values,
                    )

                    logits.append(out_logits)
                    hidden_states_offset = next_compute_range[0]

                else:
                    # Original coconut: full model for every latent pass
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, : next_compute_range[1]],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                    )

                    logits.append(outputs.logits)
                    hidden_states = outputs.hidden_states[-1]
                    kv_cache = outputs.past_key_values
                    hidden_states_offset = next_compute_range[0]

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding hidden states from layer N-2
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass: full model on remaining tokens (last latent + post-latent)
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
