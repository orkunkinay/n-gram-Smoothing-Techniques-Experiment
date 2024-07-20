# A Comparative Study of Smoothing Techniques for N-Gram Models

## Introduction
This repository contains the implementation of an experiment aimed at evaluating various smoothing techniques applied to n-gram models in Natural Language Processing (NLP). The evaluation is based on perplexity scores, which measure the effectiveness of these techniques in handling unseen events.

## Experiment Overview
The experiment involves:
- Data preprocessing and exploration
- Implementation of various smoothing techniques
  - Additive (Laplace) Smoothing
  - Good-Turing Smoothing
  - Jelinek-Mercer Smoothing
  - Kneser-Ney Smoothing
  - Modified Kneser-Ney Smoothing
- Training unigram, bigram, trigram, and 4-gram models
- Evaluation of models using perplexity as the performance metric