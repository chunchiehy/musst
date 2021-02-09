import torch
import jsonlines
import json
import argparse
import time
import requests
from tqdm import tqdm

import editdistance

from util import KMPSearch, untokenize, encode_pq

from nltk.parse import CoreNLPParser
from transformers import AlbertTokenizer

from multiprocessing import Pool


def get_spans(tokenizer, tree, span_list, text_tokens, last_end_index):
  if isinstance(tree, str):
    return (False, None)
  span = untokenize(tree.leaves())
  # Special treat for albert tokenizer
  if span in [',', '.', "'s"]:
    span_tokens = tokenizer.tokenize(span)[1:]
  # Get a list of tokens
  else:
    span_tokens = tokenizer.tokenize(span)

  after_start_index = KMPSearch(span_tokens, text_tokens[last_end_index:])
  before_start_index = KMPSearch(span_tokens, text_tokens[:last_end_index])

  start_index = -1
  if before_start_index != -1:
    start_index = before_start_index
  if after_start_index != -1:
    start_index = after_start_index + last_end_index

  if start_index != -1:
    end_index = start_index + len(span_tokens)
    span_list.append((start_index, end_index))
    return (True, end_index)
  else:
    last_end_index_tmp = last_end_index
    for subtree in tree:
      got_span, tmp = get_spans(tokenizer, subtree, span_list, text_tokens,
                                last_end_index_tmp)
      if got_span:
        last_end_index_tmp = tmp
    return (True, last_end_index_tmp)


def pruning(span_list):
  i = 0
  while i < len(span_list) - 1:
    if span_list[i][1] == span_list[i + 1][0]:
      #print("pruning!")
      span_list[i] = (span_list[i][0], span_list[i + 1][1])
      span_list.pop(i + 1)
    else:
      i = i + 1


def annotation(example):
  query_id = example['query_id']
  query = example['query']
  passage = example['passage']['passage_text']
  answer = example['answer']

  qp_text_tokens, _, _ = encode_pq(tokenizer, query, passage, max_len, task)

  # Get rid of the '.' and whitesapce
  ans = answer.strip(". ")
  ans_tokens = tokenizer.tokenize(ans)

  # Get answer span positions
  ans_span_list = []
  start_pos = KMPSearch(ans_tokens, qp_text_tokens)
  if start_pos != -1:
    # if ans are already a span in text
    # print("ans is a span in passsage")
    end_pos = start_pos + len(ans_tokens)
    ans_span_list.append((start_pos, end_pos))
    return query_id, ans_span_list, qp_text_tokens, ans, query, passage
  else:
    try:
      parse_tree = next(nlp_parser.raw_parse(ans))
      get_spans(tokenizer, parse_tree, ans_span_list, qp_text_tokens, 0)
      pruning(ans_span_list)
      return query_id, ans_span_list, qp_text_tokens, ans, query, passage
    except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError,
            ValueError, IndexError, StopIteration):
      # Bad example
      print(ans)
      print("Bad example")
      print(query_id)
      return query_id, None, None, None, None, None


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data_file", required=True, type=str)
  parser.add_argument("--output_file", required=True, type=str)
  parser.add_argument("--max_seq_len", default=256, type=int)
  parser.add_argument("--parser_url", required=True, type=str)
  parser.add_argument("--model_name", required=True, type=str)
  parser.add_argument("--task", required=True, type=str)

  args = parser.parse_args()

  nlp_parser = CoreNLPParser(url=args.parser_url)
  tokenizer = AlbertTokenizer.from_pretrained(args.model_name)
  task = args.task
  max_len = args.max_seq_len

  # Read examples
  all_example = []
  with jsonlines.open(args.train_data_file) as reader:
    for example in reader:
      all_example.append(example)
  print("The original size of dataset is ", len(all_example))

  # Now we begin flitered bad ecxamples
  flitered_examples = []
  if args.task == "qa":
    bad_examples = []
  elif args.task == "nlg":
    bad_examples = [369430, 365828]
  else:
    raise ValueError("Wrong task!")

  for example in all_example:
    query_id = example['query_id']
    if query_id in bad_examples:
      continue
    answer = example['answer'].strip(' .')
    if answer in ["", "-", ","]:
      continue
    flitered_examples.append(example)
  all_example = flitered_examples
  print("The size of dataset after flitering is ", len(all_example))

  # Doing annotations
  tic = time.time()
  if args.task == "nlg":
    p = Pool(16)
    all_ans_span_pos = p.map(annotation, all_example)
  elif args.task == "qa":
    ## Have a problem with QA
    all_ans_span_pos = []
    for example in tqdm(all_example):
      all_ans_span_pos.append(annotation(example))
  else:
    raise ValueError("Wrong task!")
  toc = time.time()
  print("Time of preprocessing is ", (toc - tic) / 60.0)

  all_annoted_spans = {}
  for (query_id, annoted_spans, qp_text_tokens, answer, query, passage) in all_ans_span_pos:
    if annoted_spans is None:
      continue
    annoted_spans_text = []
    for span_pos in annoted_spans:
      span_text = tokenizer.convert_tokens_to_string(
          qp_text_tokens[span_pos[0]:span_pos[1]])
      annoted_spans_text.append(span_text)
    result = {}
    result['annoted_spans_pos'] = annoted_spans
    result['annoted_spans_text'] = annoted_spans_text
    result['ant_answer'] = untokenize(annoted_spans_text)
    result['raw_answer'] = answer.lower()
    result['edit_distance'] = editdistance.eval(result['ant_answer'], result['raw_answer'])
    result['query'] = query
    result['passage'] = passage
    all_annoted_spans[query_id] = result

  with open(args.output_file, mode='w') as writer:
    json.dump(all_annoted_spans, writer, indent=4)