from dataset import MarcoDataset
from util import untokenize

import torch
from torch.utils.data import SequentialSampler, DataLoader

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

import os
import json
from collections import defaultdict


def predict(args, model, tokenizer, logger, data_file, set_type,
            output_pred_file):
  pred_data_file = os.path.join(args.data_dir, data_file)
  pred_dataset = MarcoDataset(data_file=pred_data_file,
                              tokenizer=tokenizer,
                              set_type=set_type,
                              task_name=args.task_name,
                              max_seq_len=args.max_seq_len,
                              max_num_spans=args.max_num_spans)

  pred_output_dir = args.output_dir
  if not os.path.exists(pred_output_dir):
    os.makedirs(pred_output_dir)

  args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)

  pred_sampler = SequentialSampler(pred_dataset)
  pred_dataloader = DataLoader(pred_dataset,
                               sampler=pred_sampler,
                               batch_size=args.pred_batch_size)

  # Pred!
  logger.info("***** Running prediction on {} set *****".format(set_type))
  logger.info("  Num examples = %d", len(pred_dataset))
  logger.info("  Batch size = %d", args.pred_batch_size)

  model.eval()

  all_candidates_ans = defaultdict(list)
  pred_iterator = tqdm(pred_dataloader, desc="Predicting", leave=False)
  for batch in pred_iterator:
    input_ids = batch['input_ids'].to(args.device)
    segment_ids = batch['segment_ids'].to(args.device)
    input_mask = batch['input_mask'].to(args.device)
    query_ids = batch['query_id']

    with torch.no_grad():
      outputs = model(input_ids=input_ids,
                      attention_mask=input_mask,
                      token_type_ids=segment_ids)
      # [batch_size, max_num_spans]
      batch_start_logits = outputs[0].detach().cpu().numpy()
      batch_end_logits = outputs[1].detach().cpu().numpy()

      num_spans = batch_start_logits.shape[1]
      batch_output_mask = input_mask.detach().cpu().numpy()
      batch_output_mask[:, -1] = 1
      batch_output_mask = (1.0 - batch_output_mask) * -10000.0
      batch_output_mask = np.expand_dims(batch_output_mask, axis=1)
      batch_output_mask = np.repeat(batch_output_mask,
                                    repeats=num_spans,
                                    axis=1)

      # We mask output logits
      batch_start_logits = batch_start_logits + batch_output_mask
      batch_end_logits = batch_end_logits + batch_output_mask

    ## generate answers
    for i in range(len(query_ids)):
      single_ans = {}
      query_id = int(query_ids[i])
      qp_ids = input_ids[i]
      all_spans_start_logits = batch_start_logits[i]
      all_spans_end_logits = batch_end_logits[i]

      if args.task_name == 'nlg':
        ans, span_pos, span_texts, score = nlg_decoding(all_spans_start_logits,
                                                 all_spans_end_logits,
                                                 tokenizer,
                                                 args.max_seq_len - 1, qp_ids)
      elif args.task_name == 'qa':
        ans, span_pos, span_texts, score = qa_decoding(all_spans_start_logits,
                                                all_spans_end_logits,
                                                tokenizer,
                                                args.max_seq_len - 1, qp_ids)
      else:
        raise ValueError("Task doesn't existe")
      
      single_ans['score'] = score
      single_ans['span_pos'] = span_pos
      single_ans['span_texts'] = span_texts
      ans = ' '.join(ans)
      ans = ans + '.'
      single_ans['candiate_answer'] = ans
      if set_type != 'test':
        single_ans['original_answer'] = pred_dataset.all_examples[query_id][
            'answer']
      single_ans['query'] = pred_dataset.all_examples[query_id]['query']
      single_ans['passage'] = pred_dataset.all_examples[query_id]['passage'][
          'passage_text']
      all_candidates_ans[query_id] = single_ans

  print()
  pred_iterator.close()
  ############################################################################
  output_f = os.path.join(pred_output_dir, "vebose_" + output_pred_file)
  with open(output_f, 'w') as writer:
    json.dump(all_candidates_ans, writer)
  ############################################################################
  output_f = os.path.join(pred_output_dir, output_pred_file)

  all_preds_ans = []
  for (query_id, single_ans) in all_candidates_ans.items():
    all_preds_ans.append({
        'query_id': query_id,
        'answers': [single_ans['candiate_answer']]
    })

  with open(output_f, 'w') as writer:
    for single_ans in all_preds_ans:
      writer.write(json.dumps(single_ans))
      writer.write("\n")
  ############################################################################

  return output_f


"""
start_logits: [max_seq_len]
end_logits: [max_seq_len]
"""


def single_span_pred_simple(start_logits, end_logits, query_id):
  start_pos = np.argmax(start_logits)
  end_pos = np.argmax(end_logits)

  return start_pos, end_pos


###### NLG decoding methods
def nlg_single_span_pred(start_logits,
                         end_logits,
                         start_mask,
                         end_mask,
                         sep_index,
                         n=10,
                         m=20):
  all_logits = {}
  extend_start_mask = (1.0 - start_mask) * -10000.0
  extend_end_mask = (1.0 - end_mask) * -10000.0

  start_probas = softmax(start_logits + extend_start_mask)
  end_probas = softmax(end_logits + extend_end_mask)
  # print(start_probas)
  start_sorted_indexs = np.argsort(start_probas)[::-1]
  end_sorted_indexs = np.argsort(end_probas)[::-1]

  # print(start_logits[start_sorted_indexs[:n]])
  # print(end_logits[end_sorted_indexs[:n]])

  for start_index in start_sorted_indexs[:n]:
    for end_index in end_sorted_indexs[:n]:
      all_logits[(
          start_index,
          end_index)] = start_probas[start_index] * end_probas[end_index]

  top_m = [
      (key, score) for (key, score) in sorted(
          all_logits.items(), key=lambda item: item[1], reverse=True)
  ][:m]
  # top_m_dict = {
  #     key: val
  #     for (key, val) in sorted(
  #         all_logits.items(), key=lambda item: item[1], reverse=True)
  # }

  best_start_index, best_end_index = top_m[0][0]
  score = top_m[0][1]
  for (indexs, score_tmp) in top_m:
    start_index, end_index = indexs
    if start_index > end_index:
      continue
    if start_index == end_index and start_index != 255:
      continue
    if start_index < 255 and end_index == 255:
      continue
    if start_index <= sep_index and end_index > sep_index:
      continue
    else:
      best_start_index, best_end_index = start_index, end_index
      score = score_tmp
      break

  return best_start_index, best_end_index, score


def nlg_decoding(all_spans_start_logits, all_spans_end_logits, tokenizer,
                 end_symbol, qp_ids):
  ans = []
  span_pos = []
  span_texts = []
  score = 0.0

  useless = [
      tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id,
      tokenizer.unk_token_id
  ]

  sep_index = qp_ids.tolist().index(tokenizer.sep_token_id)

  start_mask = np.ones(all_spans_start_logits.shape[-1])
  end_mask = np.ones(all_spans_start_logits.shape[-1])
  for (start_logits, end_logits) in zip(all_spans_start_logits,
                                        all_spans_end_logits):
    start_probas = softmax(start_logits)
    end_probas = softmax(end_logits)
    start_pred, end_pred, single_score = nlg_single_span_pred(start_logits,
                                                end_logits,
                                                start_mask=start_mask,
                                                end_mask=end_mask,
                                                sep_index=sep_index)
    start_mask[start_pred:end_pred] = 0
    end_mask[start_pred + 1:end_pred + 1] = 0
    span_pos.append((int(start_pred), int(end_pred)))
    # score += start_probas[start_pred] * end_probas[end_pred]
    score += single_score

    # We stop the prediction of spans for the "end" symbol
    if start_pred == end_symbol and end_pred == end_symbol:
      break

    # if start_pred >= end_pred, ans_span will be []
    ans_span = qp_ids[start_pred:end_pred]
    span_texts.append(tokenizer.decode(ans_span))

    # Get rid of useless span
    ans_span = [
        int(token_id) for token_id in ans_span if int(token_id) not in useless
    ]

    ans += [tokenizer.decode(ans_span)]

  score = score / len(span_pos)

  return ans, span_pos, span_texts, score


###### QA decoding methods
def qa_decoding(all_spans_start_logits, all_spans_end_logits, tokenizer,
                end_symbol, qp_ids):
  ans = []
  span_pos = []
  span_texts = []
  score = 0.0

  useless = [
      tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id,
      tokenizer.unk_token_id
  ]

  former_start = -1
  former_end = -1
  # start_mask = np.ones(all_spans_start_logits.shape[-1])
  # end_mask = np.ones(all_spans_start_logits.shape[-1])
  for (start_logits, end_logits) in zip(all_spans_start_logits,
                                        all_spans_end_logits):
    start_probas = softmax(start_logits)
    end_probas = softmax(end_logits)

    start_pred, end_pred = qa_single_span_pred(start_logits, end_logits,
                                               former_start, former_end)
    score += start_probas[start_pred] * end_probas[end_pred]
    
    former_start = start_pred
    former_end = end_pred
    # start_mask[start_pred:end_pred] = 0
    # end_mask[start_pred + 1:end_pred + 1] = 0
    span_pos.append((int(start_pred), int(end_pred)))

    # We stop the prediction of spans for the "end" symbol
    if start_pred == end_symbol and end_pred == end_symbol:
      break

    # if start_pred >= end_pred, ans_span will be []
    ans_span = qp_ids[start_pred:end_pred]
    span_texts.append(tokenizer.decode(ans_span))

    # Get rid of useless span
    ans_span = [
        int(token_id) for token_id in ans_span if int(token_id) not in useless
    ]

    ans += [tokenizer.decode(ans_span)]
  
  score = score / len(span_pos)

  return ans, span_pos, span_texts, score


def qa_single_span_pred(start_logits,
                        end_logits,
                        former_start,
                        former_end,
                        n=5,
                        m=10):
  all_logits = {}
  start_probas = softmax(start_logits)
  end_probas = softmax(end_logits)
  start_sorted_indexs = np.argsort(start_probas)[::-1]
  end_sorted_indexs = np.argsort(end_probas)[::-1]

  for start_index in start_sorted_indexs[:n]:
    for end_index in end_sorted_indexs[:n]:
      all_logits[(
          start_index,
          end_index)] = start_probas[start_index] * end_probas[end_index]

  top_m = [
      key for (key, _) in sorted(
          all_logits.items(), key=lambda item: item[1], reverse=True)
  ][:m]

  best_start_index, best_end_index = top_m[0]
  for (start_index, end_index) in top_m:
    if start_index > end_index:
      continue
    if start_index == end_index and start_index != 255:
      continue
    if start_index < 255 and end_index == 255:
      continue
    if start_index == former_start or end_index == former_end:
      continue

    best_start_index, best_end_index = start_index, end_index
    break

  return best_start_index, best_end_index