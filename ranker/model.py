from transformers import AlbertModel, AlbertPreTrainedModel

import torch.nn as nn


class RankerModel(AlbertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = 2

    self.albert = AlbertModel(config)
    self.dropout = nn.Dropout(config.classifier_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    self.init_weights()

  def forward(self,
              input_ids=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None):

    outputs = self.albert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    outputs = (logits, ) + outputs[
        2:]  # add hidden states and attention if they are here

    return outputs  # (loss), logits, (hidden_states), (attentions)