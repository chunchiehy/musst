from transformers import AlbertModel
from transformers import AlbertPreTrainedModel

import torch.nn as nn
import torch
import torch.nn.functional as F


class MUSTTransformerModel(AlbertPreTrainedModel):
  def __init__(self, config, max_num_spans, max_seq_len):
    super(MUSTTransformerModel, self).__init__(config)
    self.max_num_spans = max_num_spans
    self.max_seq_len = max_seq_len

    self.albert = AlbertModel(config)
    self.dropout = nn.Dropout(config.classifier_dropout_prob)
    self.span_outputs = nn.ModuleList(
        [nn.Linear(config.hidden_size, 2) for _ in range(max_num_spans)])
    self.relu = nn.ReLU()

    self.init_weights()

  def forward(self,
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None):
    outputs = self.albert(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds)

    # (batch_size, seq_len, hidden_size)
    transformer_output = outputs[0]
    transformer_output = self.dropout(transformer_output)
    span_start_logits = []
    span_end_logits = []
    for span_output_layer in self.span_outputs:
      # (batch_size, seq_len)
      logits = span_output_layer(transformer_output)
      # (batch_size, seq_len)
      start_logits, end_logits = torch.split(logits, 1, dim=-1)
      start_logits = start_logits.squeeze(-1)
      end_logits = end_logits.squeeze(-1)
      span_start_logits.append(start_logits)
      span_end_logits.append(end_logits)

    # (batch_size, max_num_spans, seq_len)
    span_start_logits = torch.stack(span_start_logits, dim=1)
    span_end_logits = torch.stack(span_end_logits, dim=1)

    return (
        span_start_logits,
        span_end_logits,
    ) + outputs[2:]
