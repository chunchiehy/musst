import jsonlines
import os
import json

import argparse

def select(data_file, output_dir, task, set_type, ranking_res):
  file_name = "{}_from_self_ranker.jsonl".format(set_type)
  examples = []
  with jsonlines.open(data_file) as reader:
    if set_type in ['train', 'dev']:
      for example in reader:
        if task == 'qa':
          res = (example['answers'][0] != "No Answer Present.")
        elif task == 'nlg':
          res = (example['wellFormedAnswers'] != "[]")
        else:
          raise ValueError("No such task!")
        if res:
          new_example = example.copy()
          selected_psg_id = ranking_res[str(
              new_example['query_id'])]["best_passage_index"][0]
          select_passage = new_example['passages'][selected_psg_id]
          select_passage.pop('url')
          new_example['passage'] = select_passage
          new_example['answer'] = new_example[
              'answers'] if task == 'qa' else new_example['wellFormedAnswers']
          new_example.pop('passages')
          new_example.pop('answers')
          new_example.pop('wellFormedAnswers')
          examples.append(new_example)
    elif set_type == 'test':
      for example in reader:
        new_example = {}
        selected_psg_id = ranking_res[str(
              example['query_id'])]["best_passage_index"][0]
        select_passage = example['passages'][selected_psg_id]
        select_passage.pop('url')
        new_example['passage'] = select_passage
        new_example['query'] = example['query']
        new_example['query_id'] = example['query_id']
        new_example['query_type'] = example['query_type']
        examples.append(new_example)
    else:
      raise ValueError("Dataset type doesn't exsite")

  new_data_file = os.path.join(output_dir, file_name)
  with jsonlines.open(new_data_file, mode='w') as writer:
    writer.write_all(examples)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The input data dir.")
  parser.add_argument("--ranking_res_file",
                      default=None,
                      type=str,
                      required=True,
                      help="Output file of the ranker.")
  parser.add_argument("--set_type",
                      default=None,
                      type=str,
                      required=True,
                      help="Dev or test?")
  args = parser.parse_args()  
  dataset_dir = args.data_dir
  ranking_res_file = args.ranking_res_file
  set_type  = args.set_type
  # Make dirs
  qa_dir = os.path.join(dataset_dir, 'qa')
  if not os.path.exists(qa_dir):
    os.makedirs(qa_dir)
  nlg_dir = os.path.join(dataset_dir, 'nlg')
  if not os.path.exists(nlg_dir):
    os.makedirs(nlg_dir)

  if set_type == "dev":
    data_file = os.path.join(dataset_dir, "dev_v2.1.jsonl")
  elif set_type == "test":
    data_file = os.path.join(dataset_dir, "eval_v2.1_public.jsonl")
  else:
    raise ValueError("Wrong set type")

  with open(ranking_res_file, 'r') as reader:
    ranking_res = json.load(reader)
  select(data_file, qa_dir, 'qa', set_type, ranking_res)
  select(data_file, nlg_dir, 'nlg', set_type, ranking_res)

if __name__ == "__main__":
  main()