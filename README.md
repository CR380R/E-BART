# E-BART

E-BART is a BART-based model for jointly predicting the truthfulness of statements and generating explanations. This repository brings together the original implementation and the crowdsourced human-evaluation dataset associated with two publications:

- **Model and initial study (TTO 2021):** [original Truth and Trust Online paper](https://truthandtrustonline.com/wp-content/uploads/2021/10/TTO2021_paper_16-1.pdf).
- **Extended journal study (JDIQ 2023):** [*A Neural Model to Jointly Predict and Explain Truthfulness of Statements*](https://doi.org/10.1145/3546917) (Brand et al., *Journal of Data and Information Quality*, 15(1), Article 4).

The repository root presents the JDIQ 2023 study release and reproduces the scientific data and task folders of the [published Chapter 10 thesis materials on OSF](https://osf.io/jr6vc/). The original seven-file GitHub repository is preserved under [`Historical/`](Historical/README.md), including its original README and six model files. All original model bytes are unchanged.

For questions about the original E-BART implementation, the historical README lists Erik Brand (e.brand@uqconnect.edu.au) as a contact.

## Model and Implementation

The recommended starting point is [`Source Code/E_BART_Final.ipynb`](Source%20Code/E_BART_Final.ipynb), which demonstrates the model definition, data preprocessing, training, and inference. The equivalent Python files in [`Source Code/`](Source%20Code/) are `EBART_model.py`, `custom_trainer.py`, `run_training.py`, `run_inference.py`, and `run_inference_temperature_scaling.py`.

The model builds on Hugging Face BART. Because it predicts both a truthfulness label and an explanation, the original project provides a custom `CustomTrainer` implementation for inference; the standard Hugging Face `Seq2SeqTrainer` can be used for training. The implementation targets older PyTorch and Transformers versions and is distributed without a tested environment or trained checkpoints.

### Model Training Data

The model was originally trained using [e-FEVER](https://truthandtrustonline.com/wp-content/uploads/2020/10/TTO04.pdf), which is **not** distributed in full here. For the original dataset, see its paper; the historical README also identifies its contact as dominik.stammbach@gess.ethz.ch.

The training scripts use records with fields `id`, `label`, `claim`, `retrieved_evidence`, and `summary`. The identifier tracks the example before preprocessing; `label` is the integer target, `retrieved_evidence` is the supporting evidence paragraph, and `summary` is the reference explanation. The input sequence has the form `<s> claim </s> evidence </s>`, while the generated explanation has the form `<s> explanation </s>`. Labels are handled by the classification loss rather than one-hot encoding. See the notebook and training script for the preprocessing details.

This repository also includes selected reference explanations for the 70 **crowd-evaluation statements**, which must not be confused with redistributing the full model-training corpus.

## Study Overview

The human evaluation used four conditions:

| Condition | Information shown to workers | Response dimensions |
| --- | --- | --- |
| V1 | Claim only | Truthfulness and a written justification |
| V2 | Claim and E-BART explanation | Truthfulness and a written justification |
| V3 | Claim and e-FEVER reference explanation | Truthfulness and a written justification |
| V4 | Claim and both explanations | Truthfulness, a written justification, and explanation preference |

In V4, the **reference explanation is shown first** and the **E-BART explanation second**. Thus, preference `first` (`1`) selects the reference explanation, while `second` (`2`) selects E-BART.

Each condition has **52 statements** (26 `SUPPORTS`, 26 `REFUTES`), **130 task units** with four statements each, and **520 judgments**. The release contains **2,080 judgments from 514 distinct workers**. V1–V3 assess the same 52 statements. V4 shares 34 of these and uses 18 different statements, giving **70 distinct statements** overall. The reference and E-BART explanations differ for every statement used in V4.

`SUPPORTS` and `REFUTES` are reference labels, not the worker's `True`/`False` response.

## Release Contents

| Path | Contents |
| --- | --- |
| `Source Code/` | Six original E-BART model, training, inference, and notebook files |
| `Historical/` | Byte-preserved snapshot of the original seven-file public repository (outside the JDIQ study-release inventory) |
| `Crowdsourcing Task/` | Final V1–V4 task instructions, scales, interface settings, and assignments |
| `DataFrame/` | Five connected worker-data tables |
| `Ground_Truth.csv` | 70 statements, study reference labels, and candidate official FEVER IDs |
| `Explanations.csv` | Reference e-FEVER and E-BART explanations for the 70 statements |
| `README.md`, `CHANGELOG.md`, `CITATION.cff`, `RIGHTS.md` | Documentation, citation, and reuse information |
| `release_manifest.json`, `checksums.csv` | File inventory, counts, sizes, and SHA256 checksums |

The **41-file JDIQ study-release inventory** includes 22 task JSON files, six model files, five DataFrame CSVs, two statement/explanation CSVs, and six documentation/integrity files. The additional `Historical/` snapshot and repository `.gitattributes` are outside that 41-file inventory. The copies in `Historical/` and `Source Code/` are intentionally byte-identical; they are not different versions or implementations of the model.

## Crowdsourcing Task

`Crowdsourcing Task/` contains separate `V1/`–`V4/` folders. Each provides five files: `instructions_main.json` (participant instructions), `dimensions.json` (questions, scales, and justification settings), `task.json` (interface and attribute visibility), `hits.json` (units and statement order), and `search_engine.json` (search-interface settings, **not** saved search logs). Two shared files, `instructions_dimensions.json` and `questionnaires.json`, are empty lists because those components were not used.

Each `hits.json` has **130 units**, each identified by `unit_id` and containing a four-element `documents` array. Document entries carry `id`, `gt_label`, `claim`, `gt_explanation`, and `explanation`; not all fields were visible to workers. Visibility is defined by `task.json`: V1 shows `claim`; V2 adds `explanation`; V3 adds `gt_explanation` instead; and V4 shows both as `First Explanation` (reference) and `Second Explanation` (E-BART).

The task fields `gt_explanation` and `explanation` correspond to `reference_explanation` and `ebart_explanation` in `Explanations.csv`. All four tasks use `truthfulness` with `False` = `0` and `True` = `1` and request a written justification (minimum five words). V4 additionally uses `explanation_preference` with `First` = `1` and `Second` = `2`. The reference `gt_label` is not displayed as the worker's answer.

In `workers_answers.csv`, `doc_id` matches a document's `id`, while `doc_index` gives its **zero-based position (0–3)** within the unit. The configurations contain no original worker lists, access tokens, or internal service endpoints.

## Worker Data

The five tables in `DataFrame/` combine the four conditions, identified by `task` (`V1`–`V4`), and use consistent `worker_id` pseudonyms. A dash means no corresponding source records are distributed, not necessarily that an activity did not occur.

| File | V1 | V2 | V3 | V4 | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `workers_answers.csv` | 520 | 520 | 520 | 520 | **2,080** |
| `workers_acl.csv` | 130 | 130 | 130 | 130 | **520** |
| `workers_comments.csv` | – | – | 87 | 103 | **190** |
| `workers_mturk_data.csv` | 130 | – | 129 | – | **259** |
| `workers_user_agents.csv` | 130 | 130 | 129 | – | **389** |

**Judgments.** `workers_answers.csv` has one row per assessed statement (18 columns). `judgment_id` identifies the record; `task`, `worker_id`, `batch_name`, `unit_id`, `doc_id`, and `doc_index` identify its context. `doc_truthfulness_value`/`doc_truthfulness_label` encode the worker's binary assessment (`0`/`false`, `1`/`true`). `doc_truthfulness_justification` contains **2,079 nonempty written explanations**; only `v4_0480` is missing one. `doc_accesses` and `doc_time_elapsed` record access counts and elapsed time.

V4 adds `doc_explanation_preference_value` and `doc_explanation_preference_label` (`1`/`first` = reference; `2`/`second` = E-BART). Additional response index/description fields are present for truthfulness in V3/V4 and for preference in V4. They are blank in conditions where not collected.

**Completions and comments.** `workers_acl.csv` has one record for each of the **520 selected paid task completions** (four judgments per completion). It retains attempt/configuration counters, available quality-control checks, payment-selection status, and submission times (16 columns). Source field coverage differs: for example, `current_try` is available in V1/V2, while `try_last`, `gold_checks`, and `time_submit` appear in V3/V4. Missing values do not mean failed checks. `workers_comments.csv` contains **190 nonempty completion-level comments**, separate from the per-statement justifications, with completion identifiers, `try_last`, `comment_time_submit`, and `text`.

**MTurk and technical metadata.** `workers_mturk_data.csv` contains **259 available assignment records** with task descriptors, rewards, settings, statuses, timestamps, completion times, and approval-rate information (28 columns). `workers_user_agents.csv` contains **389 browser, operating-system, and device-category records** (17 columns). Original MTurk/worker identifiers, HIT and assignment IDs, raw user-agent strings, IP addresses, precise location, device model identifiers, and private tokens are excluded.

V3 MTurk metadata cover **128 of 130 selected completions**. One completion is associated with **two historical assignment records**; both remain as separate rows with a local `assignment_record` ordinal, not a source assignment ID. The corresponding technical table also has two records for that completion, and no comparable V4 metadata are included. Do not infer which V3 assignment produced a particular answer.

**Keys and cardinalities.** The combination `task` + `worker_id` + `batch_name` + `unit_id` uniquely identifies a completion in `workers_acl.csv` and links it to **four** answers and, where present, one comment. Join judgments to ground truth and explanations on `doc_id`. Do not join only on `worker_id` or `unit_id`. The MTurk and user-agent tables can be **one-to-many** relative to completions; joining them directly to answers can duplicate judgments.

## Ground Truth and Explanations

`Ground_Truth.csv` contains the 70 `doc_id` identifiers, statement texts (`doc_claim`), study reference labels (`doc_gt_label`), and possible official FEVER IDs. Local `claim_*` identifiers are **not** official FEVER IDs. Exact-text matching against FEVER v1.0 `shared_task_dev.jsonl` gives **63 single-ID matches** and **seven statements with multiple matching IDs**. Candidate IDs appear in `fever_ids`, with label-compatible/conflicting candidates and match-status fields. In particular, `claim_43` has matching official records with conflicting native labels. Text matching does not establish a unique original source record.

`Explanations.csv` contains `reference_explanation`, `ebart_explanation`, and `explanations_identical` for every `doc_id`. **17 of 70 explanation pairs are identical**, none of them used in V4. Explanations are not reference truthfulness labels and have source-specific reuse conditions; see `RIGHTS.md`.

## Using the Release

Read identifiers as strings. From the release root, this example attaches the statements, explanations, and completion records without multiplying the **2,080** judgments:

```python
import pandas as pd

options = {"dtype": str, "keep_default_na": False}
answers = pd.read_csv("DataFrame/workers_answers.csv", **options)
claims = pd.read_csv("Ground_Truth.csv", **options)
explanations = pd.read_csv("Explanations.csv", **options)
acl = pd.read_csv("DataFrame/workers_acl.csv", **options)

data = answers.merge(claims, on="doc_id", validate="many_to_one")
data = data.merge(explanations, on="doc_id", validate="many_to_one")
data = data.merge(
    acl,
    on=["task", "worker_id", "batch_name", "unit_id"],
    validate="many_to_one",
)
assert len(data) == 2080
```

Filter `task` to analyze one condition. Keep batch, unit, and statement context when comparing judgments: a worker can appear in multiple conditions or assess the same statement in different batches.

## File Integrity, Privacy, and Reuse

`release_manifest.json` lists **39 JDIQ study-release files**, source references, expected counts, sizes, and SHA256 hashes. `checksums.csv` verifies **40 files**, including the manifest; the checksums file itself is excluded. The **41-file inventory applies to the current JDIQ release**, not the seven additional `Historical/` files or `.gitattributes`.

The 38 JDIQ files other than `README.md`, `release_manifest.json`, and `checksums.csv` are byte-identical to the version 1.0 files published on OSF. The repository-root README integrates the earlier GitHub documentation, so the two integrity files were regenerated to reflect this repository-specific documentation change. No study CSVs, task JSONs, or model files were altered.

Worker IDs are pseudonyms, **not guarantees of anonymity**. The package contains participant-authored text, timestamps, and technical metadata with potential linkage risks. Do not attempt to identify or contact participants. Use and share the participant-authored text and associated metadata with care and in accordance with applicable privacy obligations. See `RIGHTS.md` for source-specific terms; no single license covers all components.

## Citation

For the **model and its initial study**, refer to the [original TTO 2021 paper](https://truthandtrustonline.com/wp-content/uploads/2021/10/TTO2021_paper_16-1.pdf). For the **JDIQ 2023 model extension and human-evaluation dataset**, cite:

Erik Brand, Kevin Roitero, Michael Soprano, Afshin Rahimi, and Gianluca Demartini. 2023. *A Neural Model to Jointly Predict and Explain Truthfulness of Statements*. *Journal of Data and Information Quality*, 15(1), Article 4. DOI: https://doi.org/10.1145/3546917

Machine-readable citation metadata for the journal release are available in `CITATION.cff`. The exact original repository README remains at [`Historical/README.md`](Historical/README.md).
