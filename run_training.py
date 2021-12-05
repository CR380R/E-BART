# E-BART System Definition
# Author: Erik Brand, UQ
# Last Updated: 3/12/2021

# This script trains the E-BART model

from EBART_model import *

import pandas as pd
from datasets import Dataset, load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType
import nltk


###### LOAD DATA #######

# Train
df_efever = pd.read_json("/content/drive/MyDrive/Thesis/System_Development/fever_data/efever_train_set.jsonl",
                  orient="columns", lines=True)
df_efever = df_efever.set_index("id")

df_fever = pd.read_json("/content/drive/MyDrive/Thesis/System_Development/fever_data/train.jsonl",
                        orient="columns", lines=True)
df_fever = df_fever.set_index("id")

df = pd.concat([df_fever, df_efever], axis=1, join="inner")
df = df.drop(columns=['evidence', 'verifiable'])
# Convert labels to integer values
df["label"] = df["label"].replace(to_replace={'SUPPORTS':0, 'REFUTES':1, 'NOT ENOUGH INFO':2}, value=None)
# Remove + from retrieved_evidence
df["retrieved_evidence"] = df["retrieved_evidence"].str.replace("+", "")

train_dataset = Dataset.from_pandas(df)
train_dataset = train_dataset.remove_columns(['id'])

print(train_dataset)


# Valid
df_efever_val = pd.read_json("/content/drive/MyDrive/Thesis/System_Development/fever_data/efever_dev_set.jsonl",
                  orient="columns", lines=True)
df_efever_val = df_efever_val.set_index("id")

df_fever_val = pd.read_json("/content/drive/MyDrive/Thesis/System_Development/fever_data/dev.jsonl",
                        orient="columns", lines=True)
df_fever_val = df_fever_val.set_index("id")

df_val = pd.concat([df_fever_val, df_efever_val], axis=1, join="inner")
df_val = df_val.drop(columns=['evidence', 'verifiable'])
# Convert labels to integer values
df_val["label"] = df_val["label"].replace(to_replace={'SUPPORTS':0, 'REFUTES':1, 'NOT ENOUGH INFO':2}, value=None)
# Remove + from retrieved_evidence
df_val["retrieved_evidence"] = df_val["retrieved_evidence"].str.replace("+", "")

val_dataset = Dataset.from_pandas(df_val)
val_dataset = val_dataset.remove_columns(['id'])

print(val_dataset)


###### PREPROCESS DATA #######

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def preprocess(data):
  claim = data['claim']
  evidence = data['retrieved_evidence']
  summary = data['summary']
  label = data['label']

  model_inputs = tokenizer(claim, evidence, max_length=1024, truncation=True, padding=False)

  with tokenizer.as_target_tokenizer():
    summarization_labels = tokenizer(summary, max_length=128, truncation=True, padding=False)

  # Ensure padding not included in loss
  summarization_labels["input_ids"] = [
      [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in summarization_labels["input_ids"]
  ]

  model_inputs['classification_labels'] = label   # This doesn't require one-hot encoding because of the way pytorch CrossEntropy works
  model_inputs['labels'] = summarization_labels['input_ids']

  return model_inputs


train_dataset = train_dataset.map(
                preprocess,
                batched=True,
                num_proc=None,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )


val_dataset = val_dataset.map(
                preprocess,
                batched=True,
                num_proc=None,
                remove_columns=val_dataset.column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on val dataset",
            )


###### TRAINING SETUP #######

model = BartForJointPrediction.from_pretrained('facebook/bart-large')

 # Data collator
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=None,
)


training_args = Seq2SeqTrainingArguments(
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=None,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    do_eval=True,
    do_predict=False,
    do_train=True,
    eval_accumulation_steps=None,
    eval_steps=500,
    evaluation_strategy='no',
    fp16=False,
    fp16_backend='auto',
    fp16_full_eval=False,
    fp16_opt_level='O1',
    gradient_accumulation_steps=1,
    greater_is_better=None,
    group_by_length=False,
    ignore_data_skip=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=5e-05,
    length_column_name='length',
    load_best_model_at_end=False,
    local_rank=-1,
    # log_level=-1,
    # log_level_replica=-1,
    log_on_each_node=True,
    logging_dir='/tmp/tst-summarization/runs/Jul04_02-41-44_9ee3aa777e7a',
    logging_first_step=False,
    logging_steps=500,
    logging_strategy='steps',
    lr_scheduler_type='linear',
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    # mp_parameters=,
    no_cuda=False,
    num_train_epochs=3.0,
    output_dir='/tmp/tst-summarization',
    overwrite_output_dir=True,
    past_index=-1,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    predict_with_generate=True,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id='tst-summarization',
    push_to_hub_organization=None,
    push_to_hub_token=None,
    remove_unused_columns=True,
    report_to=['tensorboard'],
    resume_from_checkpoint=None,
    run_name='/tmp/tst-summarization',
    save_on_each_node=False,
    save_steps=10000,
    save_strategy='steps',
    save_total_limit=None,
    seed=42,
    sharded_ddp=[],
    skip_memory_metrics=True,
    sortish_sampler=False,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_legacy_prediction_loop=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
)



# Metric
metric = load_metric("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if True: #data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


###### TRAIN #######

train_result = trainer.train(resume_from_checkpoint=None)
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
max_train_samples = (
    len(train_dataset)
)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()