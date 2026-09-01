# NLP Authorship Attribution

Research code and experimental results for evaluating log-likelihood-based authorship detection with language models.

This repository investigates how the choice of detector model affects the ability to distinguish text from different generative sources. It accompanies the work **“Optimal Log-Likelihood Tests for Distinguishing Generative Models under Relative Entropy Constraints.”**

## Overview

The experiments evaluate sentence-level log perplexity across:

- Three domains: Wikipedia introductions, news articles, and research abstracts
- Five text sources: Human, GPT-3.5, Llama 3.1, Falcon, and DeepSeek-R1
- Four detector models: Llama 3.1, Falcon, Phi-2, and DeepSeek-R1

Detection performance is analyzed using ROC AUC, normalized log-loss separation, distributional distance, and variance diagnostics.

## Repository structure

- `analysis/` — analysis and figure-generation notebooks
- `Responses/` — sentence-level detector outputs
- `src/` — source, generated, and cleaned datasets
- `isit2026/` — ISIT 2026 paper, presentation source, and referenced figures
- `DetectLM.py` — sentence-level detection and hypothesis-testing logic
- `PerplexityEvaluator.py` — language-model log-perplexity evaluation
- `PrepareSentenceContext.py` — sentence parsing and context construction
- `many_atomic_detections.py` — batch generation of detector responses
- `load_datasets.py` — dataset loading and text-generation workflow
- `clean_responses.py` — post-processing for generated text

## Datasets

Each domain uses the following naming convention:

| File suffix | Description |
|---|---|
| `_dataset.csv` | Original dataset containing human and GPT-generated text |
| `_dataset_generated.csv` | Dataset augmented with additional model generations |
| `_dataset_clean.csv` | Final post-processed dataset with generation artifacts removed |

The included domains are:

- `abstracts`
- `news`
- `wiki`

The original datasets were loaded from Hugging Face:

- `alonkipnis/wiki-intro-long`
- `alonkipnis/news-chatgpt-long`
- `NicolaiSivesind/ChatGPT-Research-Abstracts`

## Analysis notebooks

The main figure-generation notebooks are:

- `analysis/isit_paper_figures.ipynb`
- `analysis/isit_presentation_figures.ipynb`
- `analysis/jmlr_paper_figures.ipynb`

Additional notebooks contain exploratory analysis, response-distribution comparisons, variance checks, and post-processing.

## Detector outputs

Files under `Responses/` contain sentence-level log-perplexity measurements. Their names encode:
<domain>_<source>_<context-policy>_<detector-model>.csv
Each output generally includes the sentence identifier, sentence length, detector response, and context length.

## Paper and presentation

The ISIT 2026 paper and presentation materials are available under isit2026/.
