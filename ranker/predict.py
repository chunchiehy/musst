from dataset import MarcoRankingDataset, ToTransformerInput

import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.nn import CrossEntropyLoss

import numpy as np
from tqdm import tqdm

import json
import os
from collections import defaultdict

from metric import mean_reciprocal_rank, mean_average_precision


def predict(args, model, tokenizer, logger, data_file, set_type, output_pred_file, do_eval=False):
  pred_data_file = os.path.join(args.data_dir, data_file)
  data_transformer = ToTransformerInput(set_type, tokenizer, args.max_seq_length)
  pred_dataset = MarcoRankingDataset(datafile_path=pred_data_file,
                                     dataset_type=set_type,
                                     transform=data_transformer)

  #### Don't change it #########################################################
  output_dir = args.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)

  pred_sampler = SequentialSampler(pred_dataset)
  pred_dataloader = DataLoader(pred_dataset,
                               sampler=pred_sampler,
                               batch_size=args.pred_batch_size)

  # Prediction!
  logger.info("***** Running prediction *****")
  logger.info("  Num examples = %d", len(pred_dataset))
  logger.info("  Batch size = %d", args.pred_batch_size)

  model.eval()
  ##############################################################################
  if do_eval:
    loss = 0.0
    loss_func = CrossEntropyLoss()

  logits_preds = defaultdict(dict)
  # count = 0
  for batch in tqdm(pred_dataloader, desc="Predicting"):
    # count += 1
    # if count == 10:
    #     break
    input_ids = batch['input_ids'].to(args.device)
    attention_mask = batch['attention_mask'].to(args.device)
    token_type_ids = batch['token_type_ids'].to(args.device)
    query_ids = batch['query_id'].tolist()
    passage_indexs = batch['passage_index'].tolist()
    if do_eval:
      labels = batch['is_selected'].to(args.device)
    with torch.no_grad():
      outputs = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
      # [batch_size, num_labels]
      logits = outputs[0]
      if do_eval:
        _loss = loss_func(input=logits,
                         target=labels)
        loss += _loss.item()
      logits = logits.detach().cpu().numpy()
    for (query_id, passage_index, logit) in zip(query_ids, passage_indexs, logits):
      logits_preds[query_id][passage_index] = logit

  if do_eval:
    loss = loss / len(pred_dataset)
  else:
    loss = None

  # output_pred_file = os.path.join(output_dir, 'test_logits.bin')
  # with open(output_pred_file, "w") as writer:
  #   torch.save(logits_preds, output_pred_file)
  ############################################################################
  all_rs = []
  best_passage_preds = {}
  for (query_id, logits_for_query) in logits_preds.items():
    logits_np = np.zeros(len(logits_for_query))
    for (index, logits) in logits_for_query.items():
      probas = (np.exp(logits)/np.sum(np.exp(logits)))
      logits_np[index] = probas[1]
    max_index = np.argmax(logits_np)
    res = {}
    res['best_passage_index'] = [int(max_index)]
    res['logits'] = logits_np.astype(float).tolist()
    best_passage_preds[query_id] = res
    # Try to get the ranking results
    if do_eval:
      answers = pred_dataset.all_raw_examples[query_id]['answers']
      ## Only evaluate on the QA datasets
      if answers[0] != 'No Answer Present.':
        passages_for_one_query = pred_dataset.all_raw_examples[query_id]['passages']
        is_selecteds = [passage['is_selected'] for passage in passages_for_one_query]
        zipped_pairs = zip(logits_np, is_selecteds)
        rs = [x for _, x in sorted(zipped_pairs, reverse=True)]
        all_rs.append(rs)
  print(len(all_rs))
  print("Number of precited examples is {}".format(len(best_passage_preds)))
  
  # Dump prediction
  output_f = os.path.join(output_dir, output_pred_file)
  with open(output_f, "w") as writer:
    json.dump(best_passage_preds, writer, indent=4)
  ############################################################################
  results = {}
  if do_eval:
    logger.info("***** Eval results *****")
    map = mean_average_precision(all_rs)
    mrr = mean_reciprocal_rank(all_rs)
    output_f = os.path.join(args.output_dir, "dev_eval_results.txt")
    logger.info(" Eval loss: %f", loss)
    results = {'mrr':  mrr, 'map': map}
    with open(output_f, "w") as writer:
      for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
        writer.write("%s = %s\n" % (key, str(results[key])))
  ############################################################################

  return loss, results