# E-BART System Definition
# Author: Erik Brand, UQ
# Last Updated: 3/12/2021

# This script runs inference on the E-BART model and also applies Temperature Scaling

from EBART_model import *
from custom_trainer import *

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

import pandas as pd
from datasets import Dataset, load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType
import nltk

import matplotlib.patches as mpatches
import matplotlib


###### GLOBAL VARS #######

training_len = 9999
device = 'cuda:0'


###### LOAD DATA #######

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

val_dataset = val_dataset.map(
                preprocess,
                batched=True,
                num_proc=None,
                remove_columns=val_dataset.column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on val dataset",
            )



###### EVALUATION SETUP #######

model = BartForJointPrediction.from_pretrained('../results/tst-summarization/')

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
    do_train=False,
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

trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )



###### RUN EVALUATION #######

predictions = trainer.predict(
            val_dataset, max_length=100, num_beams=4, metric_key_prefix="eval"
        )

print(predictions.metrics)



###### TEMPERATURE SCALING - PREPARATION #######

# Split validation set into training/testing subsets
training_logits = predictions.classification_predictions[:training_len]
testing_logits = predictions.classification_predictions[training_len:]

training_labels = np.array(val_dataset['classification_labels'])[:training_len]
testing_labels = np.array(val_dataset['classification_labels'])[training_len:]


# Evaluation Functions
def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE

def draw_reliability_graph(preds, labels_oneh):
  ECE, MCE = get_metrics(preds, labels_oneh)
  bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

  font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
  matplotlib.rc('font', **font)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  
  plt.savefig('calibrated_network.png', bbox_inches='tight')


###### TEMPERATURE SCALING - BEFORE #######

classification_logits = torch.from_numpy(testing_logits)
final_layer = nn.Softmax(dim=1)
class_softmax = final_layer(classification_logits)
class_softmax_flat = np.array(class_softmax).flatten()

gold_label = testing_labels
gold_label = torch.from_numpy(gold_label)
labels_oneh = torch.nn.functional.one_hot(gold_label, num_classes=3)
labels_oneh = np.array(labels_oneh).flatten()

# Save original classification, use -1 for subset of validation data used to train temperature parameter
classification_preds = torch.argmax(final_layer(classification_logits), dim=1)
classification_preds = classification_preds.numpy()
filler = np.ones(training_len) * -1
classification_preds = np.concatenate((filler, classification_preds))
df_val['pred_label_orig'] = classification_preds

# Draw reliability graph
draw_reliability_graph(class_softmax_flat, labels_oneh)


###### TEMPERATURE SCALING - AFTER #######

def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)

# TRAIN TEMPERATURE PARAMETER
temperature = nn.Parameter(torch.ones(1).cuda())
args = {'temperature': temperature}
criterion = nn.CrossEntropyLoss()

# Removing strong_wolfe line search results in jump after 50 epochs
optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

logits_list = []
labels_list = []
temps = []
losses = []

logits_list = torch.from_numpy(training_logits).to(device)
labels_list = torch.from_numpy(training_labels).to(device)

def _eval():
  loss = criterion(T_scaling(logits_list, args), labels_list)
  loss.backward()
  temps.append(temperature.item())
  losses.append(loss)
  return loss

optimizer.step(_eval)

print('Final T_scaling factor: {:.2f}'.format(temperature.item()))

plt.subplot(121)
plt.plot(list(range(len(temps))), temps)

plt.subplot(122)
plt.plot(list(range(len(losses))), losses)
plt.show()


# EVALUATION
args = {'temperature': temperature}

classification_logits = torch.from_numpy(testing_logits)
classification_logits = classification_logits.to(device)
classification_logits = T_scaling(classification_logits, args)
final_layer = nn.Softmax(dim=1)
class_softmax = final_layer(classification_logits)
class_softmax = class_softmax.cpu().detach().numpy()
class_softmax_flat = class_softmax.flatten()

gold_label = testing_labels
gold_label = torch.from_numpy(gold_label)
labels_oneh = torch.nn.functional.one_hot(gold_label, num_classes=3)
labels_oneh = np.array(labels_oneh).flatten()

# Save classification, use -1 for subset of validation data used to train temperature parameter
classification_preds = torch.from_numpy(class_softmax)
classification_preds = torch.argmax(classification_preds, dim=1)
classification_preds = classification_preds.numpy()
filler = np.ones(training_len) * -1
classification_preds = np.concatenate((filler, classification_preds))
df_val['pred_label_final'] = classification_preds

# Draw reliability graph
draw_reliability_graph(class_softmax_flat, labels_oneh)


###### ANALYSIS - EXPLANATIONS #######

summary = torch.from_numpy(predictions.predictions)
decoded_summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary]

###### SAVE RESULTS #######

df_val['pred_explanation'] = decoded_summaries
df_val.to_csv("model_eFEVER_data_eFEVER.csv", index=False)