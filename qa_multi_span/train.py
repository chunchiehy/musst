import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import os

from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import MarcoDataset
from eval import evaluate


def train(args, model, tokenizer, logger):
  """ Train the model """
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  tb_writer = SummaryWriter(log_dir=args.output_dir)

  train_data_file = os.path.join(args.data_dir, args.train_file)
  span_annotation_path = os.path.join(args.data_dir, args.span_annotation_file)
  train_dataset = MarcoDataset(data_file=train_data_file,
                               tokenizer=tokenizer,
                               set_type='train',
                               task_name=args.task_name,
                               max_seq_len=args.max_seq_len,
                               max_num_spans=args.max_num_spans,
                               span_annotation_file=span_annotation_path,
                               ed_threshold=args.ed_threshold)

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
  else:
    t_total = len(train_dataloader) * args.num_train_epochs

  # Prepare optimizer and schedule (linear warm-up and decay)
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [{
      'params': [
          p for n, p in model.named_parameters()
          if not any(nd in n for nd in no_decay)
      ],
      'weight_decay':
      args.weight_decay
  }, {
      'params': [
          p for n, p in model.named_parameters()
          if any(nd in n for nd in no_decay)
      ],
      'weight_decay':
      0.0
  }]
  optimizer = AdamW(optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    eps=args.adam_epsilon,
                    betas=(args.adam_beta1, args.adam_beta2))

  warmup_steps = int(t_total * args.warmup_rate)
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=t_total)

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d",
              args.per_gpu_train_batch_size)
  logger.info("  Total train batch size (w. parallel) = %d",
              args.train_batch_size)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  best_score = 0
  # We ignore the padding index
  loss_func = CrossEntropyLoss(ignore_index=0)
  model.train()
  model.zero_grad()
  for i in range(int(args.num_train_epochs)):
    epoch_iterator = tqdm(train_dataloader)
    for _, batch in enumerate(epoch_iterator):
      input_ids = batch['input_ids'].to(args.device)
      segment_ids = batch['segment_ids'].to(args.device)
      input_mask = batch['input_mask'].to(args.device)
      start_pos = batch['start_pos'].to(args.device)
      end_pos = batch['end_pos'].to(args.device)

      outputs = model(input_ids=input_ids,
                      attention_mask=input_mask,
                      token_type_ids=segment_ids)
      # [batch_size, max_num_spans]
      start_logits = outputs[0]

      end_logits = outputs[1]
      start_loss = loss_func(start_logits.permute(0, 2, 1), start_pos)
      end_loss = loss_func(end_logits.permute(0, 2, 1), end_pos)
      loss = (start_loss + end_loss) / 2

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      tr_loss += loss.item()

      epoch_iterator.set_description("Train loss: {:.6f}  Epoch {}/{}".format(
          loss.item(), i + 1, int(args.num_train_epochs)))

      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      model.zero_grad()
      global_step += 1

      if args.logging_steps > 0 and global_step % args.logging_steps == 0:
        # Log metrics
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('train_loss',
                             (tr_loss - logging_loss) / args.logging_steps,
                             global_step)
        logging_loss = tr_loss

      if args.evaluate_during_training and args.eval_steps > 0 and global_step % args.eval_steps == 0:
        epoch_iterator.clear()
        print('-' * 120)
        logger.info("***** Evaluating at step {} *****".format(global_step))
        # print()
        results = evaluate(args, model, tokenizer, logger)
        model.train()
        for key, value in results.items():
          tb_writer.add_scalar('dev_{}'.format(key), value, global_step)

        curr_socre = (results['rouge_l'] + results['bleu_1']) / 2
        if curr_socre > best_score:
          best_score = curr_socre
          # Save model checkpoint
          logger.info("Saving model checkpoint to %s", args.output_dir)
          model_to_save = model.module if hasattr(model, 'module') else model
          model_to_save.save_pretrained(args.output_dir)
          tokenizer.save_pretrained(args.output_dir)
          torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

      if 0 < args.max_steps < global_step:
        epoch_iterator.close()
        break
    if 0 < args.max_steps < global_step:
      break
  tb_writer.close()

  return global_step, tr_loss / global_step
