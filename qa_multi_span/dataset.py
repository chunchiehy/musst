import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import jsonlines
import nltk
from tqdm import tqdm
import editdistance

from util import untokenize, encode_pq


class MarcoDataset(Dataset):
  def __init__(self,
               data_file,
               tokenizer,
               set_type,
               task_name,
               max_seq_len,
               max_num_spans,
               span_annotation_file=None,
               ed_threshold=None):
    self.tokenizer = tokenizer
    self.set_type = set_type
    self.max_seq_len = max_seq_len
    self.max_num_spans = max_num_spans
    self.task_name = task_name

    # read dataset
    self.all_examples = {}
    with jsonlines.open(data_file) as reader:
      for example in reader:
        self.all_examples[example['query_id']] = example

    if set_type == 'train':
      print("The original dataset size is {}".format(len(self.all_examples)))

      with open(span_annotation_file, 'r') as reader:
        self.span_annotation = json.load(reader)

      # Get rid of example without annotations
      for example_id in list(self.all_examples.keys()):
        if str(example_id) not in self.span_annotation:
          self.all_examples.pop(example_id)
        else:
          self.all_examples[example_id]['ans_spans'] = self.span_annotation[
              str(example_id)]['annoted_spans_pos']

      print("The size of dataset with annotations is {}".format(
          len(self.all_examples)))

      # Now we filterted examples where annotation answer "qui est loin de la réponse origianle"
      for query_id in list(self.all_examples.keys()):
        annoted_spans_text = self.span_annotation[str(
            query_id)]['annoted_spans_text']
        reconstructed_ans = untokenize(annoted_spans_text)
        ori_ans = self.all_examples[query_id]['answer'].lower().strip(". ,")
        ds = editdistance.eval(reconstructed_ans, ori_ans)
        ## 3 seems like a good threshold !ablation study
        if ds > ed_threshold:
          self.all_examples.pop(query_id)
      print("The size of dataset with examples (Edit distance <= {}) is {}".
            format(ed_threshold, len(self.all_examples)))

      # Now just some stas:
      lens = []
      for (query_id, example) in self.all_examples.items():
        lens.append(len(example['ans_spans']))
      x, y = np.unique(lens, return_counts=True)
      for (i, j) in zip(x, y):
        print("{} : {}".format(i, j))

      # # Only examples with maximum spans and delete 0
      # for query_id in list(self.all_examples.keys()):
      #   _num_span = len(self.all_examples[query_id]['ans_spans'])
      #   if _num_span > max_num_spans or _num_span == 0:
      #     self.all_examples.pop(query_id)
      # print("The size of dataset with examples (0 < Span number <= {}) is {}".
      #       format(max_num_spans, len(self.all_examples)))

      # Get rid of examples where span is 0
      for query_id in list(self.all_examples.keys()):
        _num_span = len(self.all_examples[query_id]['ans_spans'])
        if _num_span == 0:
          self.all_examples.pop(query_id)
      print("The size of dataset with examples (0 < Span number) is {}".format(
          len(self.all_examples)))

    self.all_examples_list = list(self.all_examples.values())

  def __len__(self):
    return len(self.all_examples_list)

  def __getitem__(self, idx):
    example = self.all_examples_list[idx]
    example = self.to_albert_input(example)
    return example

  def to_albert_input(self, example):
    ## Just for test QA
    query = example['query']
    passage = example['passage']['passage_text']

    qp_text_tokens, p_tokens, q_tokens = encode_pq(self.tokenizer, query,
                                                   passage, self.max_seq_len,
                                                   self.task_name)

    pq_token_ids = self.tokenizer.convert_tokens_to_ids(qp_text_tokens)
    pq_type_ids = (len(q_tokens) + 2) * [0] + (len(p_tokens) + 1) * [1]
    assert len(pq_token_ids) == len(pq_type_ids)
    pq_mask = [1] * len(pq_token_ids)

    # Zero-pad up to the sequence length.
    padding_length = self.max_seq_len - len(pq_token_ids)
    pq_token_ids = pq_token_ids + ([self.tokenizer.pad_token_id] *
                                   padding_length)
    pq_mask = pq_mask + ([0] * padding_length)
    pq_type_ids = pq_type_ids + ([0] * padding_length)

    qp_text_tokens += [self.tokenizer.pad_token] * padding_length
    assert len(pq_token_ids) == self.max_seq_len
    assert len(pq_mask) == self.max_seq_len
    assert len(pq_type_ids) == self.max_seq_len
    assert len(qp_text_tokens) == self.max_seq_len

    input_ids = torch.tensor(pq_token_ids, dtype=torch.long)
    input_mask = torch.tensor(pq_mask, dtype=torch.long)
    segment_ids = torch.tensor(pq_type_ids, dtype=torch.long)

    # Ans span parts
    if self.set_type == "train":
      ans_spans = example['ans_spans']
      if len(ans_spans) > self.max_num_spans:
        ans_spans = ans_spans[:self.max_num_spans]
      # The ending symbol is (self.max_len-1)
      ans_spans = ans_spans + [(self.max_seq_len - 1, self.max_seq_len - 1)]
      # We are padding ans span now (0, 0)
      ans_pad_len = self.max_num_spans + 1 - len(ans_spans)
      ans_spans = ans_spans + [(0, 0)] * ans_pad_len
      start_pos = [spans[0] for spans in ans_spans]
      end_pos = [spans[1] for spans in ans_spans]

      start_pos = np.array(start_pos)
      end_pos = np.array(end_pos)
      start_pos = torch.tensor(start_pos, dtype=torch.long)
      end_pos = torch.tensor(end_pos, dtype=torch.long)

      example = {
          'query_id': example['query_id'],
          'input_ids': input_ids,
          'input_mask': input_mask,
          'segment_ids': segment_ids,
          'start_pos': start_pos,
          'end_pos': end_pos
      }
    else:
      example = {
          'query_id': example['query_id'],
          'input_ids': input_ids,
          'input_mask': input_mask,
          'segment_ids': segment_ids
      }
    return example
