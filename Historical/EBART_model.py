# E-BART System Definition
# Author: Erik Brand, UQ
# Last Updated: 3/12/2021

# This script defines the E-BART model

from transformers import BartTokenizer, BartModel, BartPretrainedModel, BartConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.models.bart.modeling_bart import BartClassificationHead, shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation_logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput
from transformers.file_utils import ModelOutput

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import numpy as np



class Seq2SeqJointOutput(ModelOutput):
  classification_logits: torch.FloatTensor = None
  loss: Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
  decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
  cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
  encoder_last_hidden_state: Optional[torch.FloatTensor] = None
  encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None



class BartForJointPrediction(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,  # Defaults to 3 in BART config
            config.classifier_dropout,
        )

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        classification_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # """
        # labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        #     Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
        #     config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
        #     (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        # Returns:
        # """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Summarization Logits
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias


        # Classification Logits
        hidden_states = outputs[0]  # last hidden state

        classification_logits = None
        if input_ids is not None:   # In all cases except generating autoregressively - we don't want to produce classficiation logits
            # Can't just use decoder_input_ids all the time as these are already shifted right - might not have ending eos_token_id <s>
            if labels is not None:
                # Training
                eos_mask = labels.eq(self.config.eos_token_id)
            else:
                # Inference
                # We want this to match what happens in the training task: 
                # labels are shifted right and passed to the decoder
                # The decoder predicts eos_token_id based on the last token in label (that is not eos_token_id as this has been shifted off)
                # For inference we pass in final predicted summary, make classification prediction based on last token before eos_token_id, just like training
                eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
                # Shift mask left so that True lines up with token immediately before eos_token_id <s>
                shifted_mask = eos_mask.new_zeros(eos_mask.shape)
                shifted_mask[:, :-1] = eos_mask[:, 1:].clone()
                shifted_mask[:, -1] = False
                eos_mask = shifted_mask
            
            
            if len(torch.unique(eos_mask.sum(1))) > 1:
                # print(decoder_input_ids)
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                :, -1, :
            ]
            classification_logits = self.classification_head(sentence_representation)


        # Summarization Loss
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Classification Loss
        classification_loss = None
        if classification_labels is not None:
            if self.config.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                classification_loss = loss_fct(classification_logits.view(-1), classification_labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                classification_loss = loss_fct(classification_logits.view(-1, self.config.num_labels), classification_labels.view(-1))

        loss = None
        if (labels is not None) and (classification_labels is not None):
            # Joint loss
            loss = 0.5 * masked_lm_loss + 0.5 * classification_loss
        elif labels is not None:
            # Only summarisation loss
            loss = masked_lm_loss
        elif classification_labels is not None:
            # Only classification loss
            loss = classification_loss

        if not return_dict:
            output = (lm_logits) + outputs[1:] + (classification_logits)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqJointOutput(
            loss=loss,
            logits=lm_logits,
            classification_logits=classification_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
    


    # Override greedy_search from generate MixIn:
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        
        # First (autoregressive) pass of model to generate summary
        summarization_outputs = super().generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            max_time=max_time,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=True,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values,
            synced_gpus=synced_gpus,
            **model_kwargs)
        
        # Second pass of model to get classification output
        classification_outputs = self(input_ids=input_ids,    # Do another pass of the encoder
                                      decoder_input_ids=summarization_outputs['sequences'], # The final summary
                                      use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states,
                                      **model_kwargs)
        
        return (classification_outputs['classification_logits'], summarization_outputs['sequences'])
