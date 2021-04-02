import json
import os
import time
import numpy as np

import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The input data dir.")
  parser.add_argument("--task",
                      default=None,
                      type=str,
                      required=True,
                      help="Task name: qa or nlg.")
  args = parser.parse_args()  
  data_dir = args.data_dir
  task = args.task

  # Read dataset
  with open(os.path.join(data_dir, 'dev_v2.1.json'), 'r',
            encoding='utf-8') as reader:
    data_dict = json.load(reader)
    query_ids = data_dict['query_id']
    passages = data_dict['passages']
    well_formed_answers = data_dict['wellFormedAnswers']
    answers = data_dict['answers']

    if task == "nlg":
      # Get NLG examples
      nlg_query_ids = query_ids.copy()
      for query_id in query_ids:
        if well_formed_answers[query_id] == "[]":
          nlg_query_ids.pop(query_id)
      query_ids = nlg_query_ids
    elif task == "qa":
      # Get QA examples
      qa_query_ids = query_ids.copy()
      for query_id in query_ids:
        if answers[query_id][0] == "No Answer Present.":
          qa_query_ids.pop(query_id)
      query_ids = qa_query_ids

    print("The dataset size is {}".format(len(query_ids)))

    all_examples = []
    for query_id in query_ids:
      ans = well_formed_answers[query_id] if task == 'nlg' else answers[query_id]
      single_example = {'query_id': query_ids[query_id], 'answers': ans}
      all_examples.append(single_example)
    dev_ref_file = os.path.join(data_dir, task, 'dev_ref.json')
    with open(dev_ref_file, 'w') as writer:
      for single_ans in all_examples:
        writer.write(json.dumps(single_ans))
        writer.write("\n")

if __name__ == "__main__":
  main()