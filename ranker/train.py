import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from torchvision import transforms

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import os

from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import MarcoRankingDataset, ToTransformerInput
from predict import predict


def train(args, model, tokenizer, logger):
  """ Train the model """
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  tb_writer = SummaryWriter(log_dir=args.output_dir)

  # Prepare the dataset
  train_data_file = os.path.join(args.data_dir, 'train_v2.1.jsonl')
  data_transformer = ToTransformerInput('train',
                      tokenizer,
                      args.max_seq_length)
  train_dataset = MarcoRankingDataset(datafile_path=train_data_file,
                    dataset_type='train',
                    transform=data_transformer)

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset,
                  sampler=train_sampler,
                  batch_size=args.train_batch_size)

  # Calculating training step
  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
  else:
    t_total = len(train_dataloader) * args.num_train_epochs

  # Prepare optimizer and schedule (linear warm-up and decay)
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
  optimizer = AdamW(optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=(args.adam_beta1, args.adam_beta2))

  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info("  Total train batch size (w. parallel) = %d", args.train_batch_size)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  best_score = 0.0
  loss_func = CrossEntropyLoss()
  model.train()
  model.zero_grad()
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
  for _ in train_iterator:
    ######################################################################
    train_dataset.resampling()
    print(train_dataset.all_examples[:20])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                    sampler=train_sampler,
                    batch_size=args.train_batch_size)
    #######################################################################
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for _, batch in enumerate(epoch_iterator):
      input_ids = batch['input_ids'].to(args.device)
      attention_mask = batch['attention_mask'].to(args.device)
      token_type_ids = batch['token_type_ids'].to(args.device)
      labels = batch['is_selected'].to(args.device)

      outputs = model(input_ids=input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids)

      # [batch_size, num_labels]
      logits = outputs[0]

      loss = loss_func(input=logits,
               target=labels)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      tr_loss += loss.item()

      pred_indexs = np.argmax(logits.detach().cpu().numpy(), axis=-1)
      acc = np.sum(pred_indexs == labels.detach().cpu().numpy()) / args.train_batch_size
      epoch_iterator.set_description("Train loss: % 12.6f, Acc: % 12.6f" % (loss.item(), acc))

      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      model.zero_grad()
      global_step += 1

      if args.logging_steps > 0 and global_step % args.logging_steps == 0:
        # Log metrics
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('train_loss',
                   (tr_loss - logging_loss) / args.logging_steps, global_step)
        tb_writer.add_scalar('train_acc', acc, global_step)
        logging_loss = tr_loss

      if args.eval_steps > 0 and global_step % args.eval_steps == 0:
        logger.info("***** Running evaluation on dev set at step %s *****", global_step)
        dev_loss, results = predict(args, model, tokenizer, logger,
                                    args.eval_file, 'dev',
                                    'dev_best_passage_pred.json', do_eval=True)
        tb_writer.add_scalar('dev_loss', dev_loss, global_step)
        for key, value in results.items():
          tb_writer.add_scalar('{}'.format(key), value, global_step)

        curr_socre = (results["mrr"] + results["map"]) / 2
        if curr_socre > best_score:
          best_score = curr_socre
          # Save model checkpoint
          # Take care of distributed/parallel training
          model_to_save = model.module if hasattr(model, 'module') else model
          model_to_save.save_pretrained(args.output_dir)
          tokenizer.save_pretrained(args.output_dir)
          torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
          logger.info("Saving model checkpoint to %s", args.output_dir)

      if 0 < args.max_steps < global_step:
        epoch_iterator.close()
        break
    if 0 < args.max_steps < global_step:
      train_iterator.close()
      break
  tb_writer.close()

  logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)
