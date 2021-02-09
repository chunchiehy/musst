import torch
from torch.utils.data import Dataset
import json
import jsonlines
import random

class MarcoRankingDataset(Dataset):
  def __init__(self, datafile_path, dataset_type, transform):
    self.all_raw_examples = {}
    with jsonlines.open(datafile_path) as reader:
      for example in reader:
        self.all_raw_examples[example['query_id']] = example

    self.all_examples = []
    if dataset_type == 'train':
      self.all_examples = _resampling(self.all_raw_examples)
    elif dataset_type == 'dev':
      count = 0
      for raw_example in self.all_raw_examples.values():
        query_id = raw_example['query_id']
        passages = raw_example['passages']
        query_text = raw_example['query']
        count += 1
        for (i, single_passage) in enumerate(passages):
          self.all_examples.append(
            {'query_id': query_id,
             'passage_index': i,
             'query': query_text,
             'passage': single_passage['passage_text'],
             'is_selected': single_passage['is_selected']})
      print("count: ", count)
    elif dataset_type == 'test':
      # Test dataset
      count = 0
      for raw_example in self.all_raw_examples.values():
        query_id = raw_example['query_id']
        passages = raw_example['passages']
        query_text = raw_example['query']
        count += 1
        for (i, single_passage) in enumerate(passages):
          self.all_examples.append({
              'query_id': query_id,
              'passage_index': i,
              'query': query_text,
              'passage': single_passage['passage_text']})
      print("count: ", count)
    else:
      raise ValueError("Dataset type doesn't exist")
    self.transform = transform

  def __len__(self):
    return len(self.all_examples)

  def __getitem__(self, idx):
    example = self.all_examples[idx]

    if self.transform:
      example = self.transform(example)

    return example

  def resampling(self):
    self.all_examples = _resampling(self.all_raw_examples)


def _resampling(all_raw_examples):
  all_examples = []
  for raw_example in all_raw_examples.values():
    query_id = raw_example['query_id']
    passages = raw_example['passages']
    query_text = raw_example['query']

    is_selecteds = []
    not_selecteds = []
    for (i, single_passage) in enumerate(passages):
      if single_passage['is_selected'] == 1:
        is_selecteds.append(i)
      else:
        not_selecteds.append(i)
    for selected_index in is_selecteds:
      if len(not_selecteds) == 0:
        break
      not_selected_index = random.choice(not_selecteds)

      all_examples.append({
          'query_id': query_id,
          'query': query_text,
          'passage': passages[selected_index]['passage_text'],
          'passage_index': selected_index,
          'is_selected': 1})
      all_examples.append({
          'query_id': query_id,
          'query': query_text,
          'passage': passages[not_selected_index]['passage_text'],
          'passage_index': not_selected_index,
          'is_selected': 0})
      not_selecteds.remove(not_selected_index)
  return all_examples


class ToTransformerInput(object):
  def __init__(self, dataset_type, tokenizer, max_seq_len):
    self.dataset_type = dataset_type
    self.max_len = max_seq_len
    self.tokenizer = tokenizer

  def __call__(self, example):
    query_id = example['query_id']
    passage_index = example['passage_index']
    query = example['query']
    passage = example['passage']

    encode_res = self.tokenizer.encode_plus(text=query,
                        text_pair=passage,
                        max_length=self.max_len,
                        pad_to_max_length=True)

    input_ids = torch.tensor(encode_res['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(encode_res['attention_mask'], dtype=torch.long)
    token_type_ids = torch.tensor(encode_res['token_type_ids'], dtype=torch.long)

    if self.dataset_type != 'test':
      is_selected = example['is_selected']
      is_selected = torch.tensor(is_selected, dtype=torch.long)
      example = {'query_id': query_id,
             'passage_index': passage_index,
             'input_ids': input_ids,
             'attention_mask': attention_mask,
             'token_type_ids': token_type_ids,
             'is_selected': is_selected}
    else:
      example = {'query_id': query_id,
             'passage_index': passage_index,
             'input_ids': input_ids,
             'attention_mask': attention_mask,
             'token_type_ids': token_type_ids}
    return example