# Multi-span Style Extraction for Generative Reading Comprehension
Code for the paper:
[Multi-span Style Extraction for Generative Reading Comprehension](https://arxiv.org/abs/2009.07382)  
Junjie Yang, Zhuosheng Zhang, Hai Zhao
![framework](framework.png)

## Dependencies
The code was tested with `python 3.7` and `pytorch 1.2.0`. If you use conda, you can create an env and install them as follows:
```
conda create --name marco python==3.7
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Install standford corenlp: https://stanfordnlp.github.io/CoreNLP/

## Datasets
You can download MS Marco v2.1 dataset here: 
https://microsoft.github.io/msmarco/. Put all the files into a data directory.

Now preprocess the datasets:
```bash
cd preprocessing
# Spilt 'qa' and 'nlg' subsets
python dataset_spilt.py --data_dir=${data_dir}

# Create dev reference files for evaluation
python create_dev_ref.py --data_dir=${data_dir} --task='qa'
python create_dev_ref.py --data_dir=${data_dir} --task='nlg'
```

## Ranking
Marco provides mutiple reading passages for each question, so before answering question, we need to select the most relevant one.

Train a ranker on 4 GPUs:
```bash
cd ../ranker
python main.py \
    --model_name_or_path='albert-base-v2' \
    --data_dir=${data_dir} \
    --output_dir=${expr_dir}/ranker
    --do_train \
    --learning_rate=1e-05 \
    --num_train_epochs=3.0 \
    --warmup_steps=2497 \
    --per_gpu_train_batch_size=32 \
    --eval_steps=8324 \
    --logging_steps=100 \
    --max_seq_length=256 \
    --seed=96 \
    --weight_decay=0.01
```
Eval the trained model on dev set:
```bash
export TRAINED_MODEL=${expr_dir}/ranker
python main.py \
    --data_dir=${data_dir}  \
    --model_name_or_path=$TRAINED_MODEL \
    --output_dir=$TRAINED_MODEL/res  \
    --max_seq_len=256 \
    --do_eval \
    --per_gpu_pred_batch_size=128  
```
You will find the evaluation results in a file named `dev_eval_results.txt` in the `output_dir`:
```
map = 0.7109500464755606
mrr = 0.715590335907891
```
Now select the most relevant passages with our ranker on dev set:
```bash
python select_best_passage.py \
    --data_dir=${data_dir} \
    --ranking_res_file=${expr_dir}/ranker/dev_best_passage_pred.json \
    --set_type=dev
```
This will generate a file named as `dev_from_self_ranker.jsonl`.


> The following scripts run the experiments on NLG subset, for QA subset, you just need to change argument `task` or `task_name` from "nlg" to "qa".
## Syntactic multi-span answer annotator
Now we need to transform the original answers in the training set to annotated spans.

Start Stanford CoreNLP Parser server:
```bash
java -mx20g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -preload tokenize,ssplit,pos,parse \
    -port 8889 
```
Get annotated answer spans:
```bash
cd qa_multi_span
python annotator.py \
    --train_data_file=... \
    --model_name="albert-xlarge-v2" \
    --parser_url="http://localhost:8889" \
    --set_type="train" \
    --task="nlg" \
    --output_file="train_span_annotation.json"
```

## Question-answering module
Training:
```bash
python main.py \
    --model_name_or_path=albert-xxlarge-v2 \
    --do_train \
    --data_dir=${data_dir}  \
    --output_dir=${expr_dir} \
    --eval_file=dev_from_self_ranker.jsonl \
    --span_annotation_file="train_span_annotation.json" \
    --overwrite_output_dir  \
    --per_gpu_train_batch_size=8   \
    --per_gpu_pred_batch_size=8  \
    --num_train_epochs=5.0 \
    --learning_rate=3e-5 \
    --evaluate_during_training \
    --eval_steps=1846\
    --max_seq_len=256 \
    --max_num_spans=9 \
    --ed_threshold=8\
    --task_name=nlg \
    --seed=1996
```

Evaluate on dev set with passages selected by our trained ranker:
```bash
python main.py \
    --model_name_or_path=${expr_dir} \
    --output_dir=${expr_dir}/dev_with_ranker  \
    --do_eval \
    --data_dir=${data_dir}  \
    --eval_file=dev_from_self_ranker.jsonl \
    --per_gpu_pred_batch_size=32   \
    --max_seq_len=512 \
    --max_num_spans=9\
    --reference_file=dev_ref.json \
    --task_name=nlg
```

You will get a result file named as `dev_eval_results.txt` in the `output_dir`:
```
 Dev F1 = 1.0
 Dev bleu_1 = 0.642251479046639
 Dev bleu_2 = 0.5687728163471556
 Dev bleu_3 = 0.5226526089835294
 Dev bleu_4 = 0.4883366779450553
 Dev rouge_l = 0.6624096330217961
```
The predictions are in the file `dev_prediction.json` and `vebose_dev_prediction.json`. Here are some prediction examples:
```json
{
  "15177": {
    "score": 0.6119573291265018,
    "span_pos": [
      [4, 5],
      [21, 22],
      [6, 9],
      [29, 30],
      [22, 26],
      [255, 255]
    ],
    "span_texts": ["population", "of", "albany, minnesota", "is", "2,662"],
    "candiate_answer": "population of albany, minnesota is 2,662.",
    "original_answer": ["The population of Albany, Minnesota is 2,662. "],
    "query": "albany mn population",
    "passage": "Albany, Minnesota, as per 2017 US Census estimate, has a community population of 2,662 people. Albany is located in Stearns County, 20 miles west of St. Cloud and 80 miles northwest of Minneapolis/St. Paul on Interstate 94 (I-94). Albany has direct access to State Highway 238, which originates in Albany."
  },
  "114414": {
    "score": 0.4104470265124703,
    "span_pos": [
      [1, 6],
      [27, 35],
      [255, 255]
    ],
    "span_texts": ["current weather in volcano,", "is 48 degrees and patchy rain possible"],
    "candiate_answer": "current weather in volcano, is 48 degrees and patchy rain possible.",
    "original_answer": ["The Volcano forecast for Apr 12 is 52 degrees and Patchy light rain."],
    "query": "current weather in volcano, ca",
    "passage": "Hourly Forecast Detailed. 1  0am:The Volcano, CA forecast for Apr 03 is 48 degrees and Patchy rain possible. 2  3am:The Volcano, CA forecast for Apr 03 is 44 degrees and Clear. 3  6am:The Volcano, CA forecast for Apr 03 is 41 degrees and Clear.  9am:The Volcano, CA forecast for Apr 03 is 48 degrees and Sunny."
  },
  "9083": {
    "score": 0.6377308023817662,
    "span_pos": [
      [14, 24],
      [255, 255]
    ],
    "span_texts": ["hippocrates is considered the father of modern medicine"],
    "candiate_answer": "hippocrates is considered the father of modern medicine.",
    "original_answer": ["Hippocrates is considered the father of modern medicine."],
    "query": "____________________ is considered the father of modern medicine.",
    "passage": "TRUE. Hippocrates is considered the father of modern medicine because he did not believe that illness was a punishment inflicted by the gods. True False. Weegy: TRUE. [ "
  }
}
```

## License 
MIT