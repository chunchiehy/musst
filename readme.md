# Multi-span Style Extraction for Generative Reading Comprehension
Code for the paper:
[Multi-span Style Extraction for Generative Reading Comprehension](https://arxiv.org/abs/2009.07382)  
Junjie Yang, Zhuosheng Zhang, Hai Zhao

## Dependencies
The code was tested in `python3.7`.
Install the required packages:
```bash
pip install -r requirements.txt
```

## Passage ranker
Train the passage ranker:



## Syntactic multi-span answer annotator
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
Evaluate on dev set:
# Eval in dev with single ranker
python main.py \
    --model_name_or_path=${expr_dir} \
    --output_dir=${expr_dir}/dev_single_ranker  \
    --do_eval \
    --data_dir=${data_dir}  \
    --eval_file=dev_from_self_ranker.jsonl \
    --per_gpu_pred_batch_size=32   \
    --max_seq_len=256 \
    --max_num_spans=9\
    --reference_file=dev_ref.json \
    --task_name=nlg

## License 
MIT