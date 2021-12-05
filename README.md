# E-BART
E-BART model for jointly predicting and explaining truthfulness


## Introduction

This repository holds the code for the E-BART model. The associated paper can be found at the [following link](https://truthandtrustonline.com/wp-content/uploads/2021/10/TTO2021_paper_16-1.pdf). 

Please contact Erik Brand (e.brand@uqconnect.edu.au) for any queries.

## Repository Overview

The best place to start is E_BART_Final.ipynb. This Jupyter notebook provides the model definition, as well as examples of how to load data, train the model, and run inference on the model.

Python scripts for the model definition, running training, and running inference are also provided, however the Jupyter notebook is the preferred method of using the model.

## Intricacies

The code was written to conform to the HuggingFace style, and indeed the model inherits from the base HuggingFace BART implementation. However, due to the unique 'joint prediction' nature of the model, a custom HuggingFace Trainer was developed, based on the standard Trainer library. The model can be trained using the standard HuggingFace 'Seq2SeqTrainer,' but running inference requires the 'CustomTrainer' implemented in this repository.

## Data

This model was originally trained on the e-FEVER dataset. To access the dataset, please see the original [paper](https://truthandtrustonline.com/wp-content/uploads/2020/10/TTO04.pdf) and contact the author: dominik.stammbach@gess.ethz.ch.

The model can be used with other datasets as long as they follow the specified format below.

The training script demonstrates how the data should be preprocessed. Ideally, the data should be in a pandas dataframe with the following format: `id, label, claim, retrieved_evidence, summary`.
Where:
- id is simply used to keep track of each example. This is removed when preprocessing the data;
- label is an integer ground-truth label corresponding to the correct classification of the example;
- claim is a string representing the claim/query to verify;
- retrieved evidence is a single string representing the paragraph of evidence against which to verify the claim; and
- summary is a string representing the ground-truth explanation.

The preprocessor tokenises the strings, and formats the input/output pairings. The resulting formatted input is: `<s> claim </s> evidence </s>`. The output consists of two components, the label (NOT one-hot encoded as this is taken care of by the PyTorch CrossEntropyLoss), and the ground-truth explanation: `<s> explanation </s>`. `<s>` is the BART start token, and `</s>` is the sequence separator token.