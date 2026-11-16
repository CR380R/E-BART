# E-BART System Definition
# Author: Erik Brand, UQ
# Last Updated: 3/12/2021

# This script defines the custom Huggingface Trainer used to evaluate the E-BART model

from transformers import Trainer
from transformers.utils import logging
from typing import NamedTuple
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    SequentialDistributedSampler,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    speed_metrics,
)
from transformers.debug_utils import DebugOption

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

import time
import math
import collections
import nltk

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


nltk.download("punkt", quiet=True)

logger = logging.get_logger(__name__)


class JointPredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    classification_predictions: Union[np.ndarray, Tuple[np.ndarray]]
    classification_label_ids: Optional[np.ndarray]


class JointEvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    classification_predictions: Union[np.ndarray, Tuple[np.ndarray]]
    classification_label_ids: Optional[np.ndarray]


class CustomTrainer(Trainer):
    def prediction_step(
          self,
          model: nn.Module,
          inputs: Dict[str, Union[torch.Tensor, Any]],
          prediction_loss_only: bool,
          ignore_keys: Optional[List[str]] = None,
      ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
          """
          Perform an evaluation step on :obj:`model` using obj:`inputs`.

          Subclass and override to inject custom behavior.

          Args:
              model (:obj:`nn.Module`):
                  The model to evaluate.
              inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                  The inputs and targets of the model.

                  The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                  argument :obj:`labels`. Check your model's documentation for all accepted arguments.
              prediction_loss_only (:obj:`bool`):
                  Whether or not to return the loss only.

          Return:
              Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
              labels (each being optional).
          """

          if not self.args.predict_with_generate or prediction_loss_only:
              return super().prediction_step(
                  model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
              )

          has_labels = "labels" in inputs
          inputs = self._prepare_inputs(inputs)

          # XXX: adapt synced_gpus for fairscale as well
          gen_kwargs = {
              "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
              "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
              "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
          }

          result = self.model.generate(
              inputs["input_ids"],
              attention_mask=inputs["attention_mask"],
              **gen_kwargs,
          )

          classification_logits = result[0]
          generated_tokens = result[1]

          # in case the batch is shorter than max length, the output should be padded
          if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
              generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

          with torch.no_grad():
              if self.use_amp:
                  with autocast():
                      outputs = model(**inputs)
              else:
                  outputs = model(**inputs)
              if has_labels:
                  if self.label_smoother is not None:
                      loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                  else:
                      loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
              else:
                  loss = None

          if self.args.prediction_loss_only:
              return (loss, None, None)

          labels = inputs["labels"]
          classification_labels = inputs["classification_labels"]
          if labels.shape[-1] < gen_kwargs["max_length"]:
              labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

          return (loss, generated_tokens, labels, classification_logits, classification_labels)



    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> JointEvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        classification_preds_host = None
        classification_labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_preds_classification = None
        all_labels_classification = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels, classification_logits, classification_labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if classification_logits is not None:
                classification_logits = self._pad_across_processes(classification_logits)
                classification_logits = self._nested_gather(classification_logits)
                classification_preds_host = classification_logits if classification_preds_host is None else nested_concat(classification_preds_host, classification_logits, padding_index=-100)
            if classification_labels is not None:
                classification_labels = self._pad_across_processes(classification_labels)
                classification_labels = self._nested_gather(classification_labels)
                classification_labels_host = classification_labels if classification_labels_host is None else nested_concat(classification_labels_host, classification_labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if classification_preds_host is not None:
                    classification_logits = nested_numpify(classification_preds_host)
                    all_preds_classification = classification_logits if all_preds_classification is None else nested_concat(all_preds_classification, classification_logits, padding_index=-100)
                if classification_labels_host is not None:
                    classification_labels = nested_numpify(classification_labels_host)
                    all_labels_classification = (
                        classification_labels if all_labels_classification is None else nested_concat(all_labels_classification, classification_labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, classification_preds_host, classification_labels_host = None, None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if classification_preds_host is not None:
            classification_logits = nested_numpify(classification_preds_host)
            all_preds_classification = classification_logits if all_preds_classification is None else nested_concat(all_preds_classification, classification_logits, padding_index=-100)
        if classification_labels_host is not None:
            classification_labels = nested_numpify(classification_labels_host)
            all_labels_classification = classification_labels if all_labels_classification is None else nested_concat(all_labels_classification, classification_labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_preds_classification is not None:
            all_preds_classification = nested_truncate(all_preds_classification, num_samples)
        if all_labels_classification is not None:
            all_labels_classification = nested_truncate(all_labels_classification, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return JointEvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples, classification_predictions=all_preds_classification, classification_label_ids=all_labels_classification)



    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # From seq2seqTrainer:
        self._max_length = max_length
        self._num_beams = num_beams

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics





    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> JointPredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # From seq2seqTrainer:
        self._max_length = max_length
        self._num_beams = num_beams

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return JointPredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics, classification_predictions=output.classification_predictions, classification_label_ids=output.classification_label_ids)



    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor