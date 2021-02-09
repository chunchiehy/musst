import os

from predict import predict

from official_evaluation.ms_marco_eval import compute_metrics_from_files
MAX_BLEU_ORDER = 4


def evaluate(args, model, tokenizer, logger):
  logger.info("***** Running evaluation on dev set *****")
  dev_prediction_file = predict(args, model, tokenizer, logger, args.eval_file,
                                "dev", "dev_prediction.json")
  reference_file = os.path.join(args.data_dir, args.reference_file)

  logger.info("***** Eval results *****")
  metrics_res = compute_metrics_from_files(reference_file, dev_prediction_file,
                                           MAX_BLEU_ORDER)
  for metric in sorted(metrics_res):
    logger.info(" Dev %s: %s", metric, str(metrics_res[metric]))

  eval_results_file = os.path.join(args.output_dir, "dev_eval_results.txt")
  with open(eval_results_file, "w") as writer:
    for metric in sorted(metrics_res):
      writer.write(" Dev %s = %s\n" % (metric, str(metrics_res[metric])))

  return metrics_res