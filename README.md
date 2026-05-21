# Distillation on open-source LLMs for English to Occitan NMT

This repo hosts an initial version of an end-to-end pipeline for translating English sentences to Occitan using a Mistral-7B-v0.3 model distilled using sequence KD into a TinyLlama-1.1B-Chat-v1.0 model.
The model fine-tuning and knowledge distillation utilises the CCMatrix English-Occitan parallel corpus of 1.8M sentences, and the performance has been evaluated using a
held-out set of FLORES-200, alongside a dedicated Marian MT baseline for comparison. These have also been compared against the zero-shot chat versions of my model setup to demonstrate the performance gains from finetuning on a hard task like NMT on a low-resource langauge.

## Motivation

### Introduction
This personal project was primarily an attempt to understand knowledge distillation in LLMs and their applications for machine translation. The initial idea came from [Enis and Hopkins (2024), "From LLM to NMT: Advancing Low-Resource Machine Translation with Claude"](https://arxiv.org/abs/2404.13813), which showed that knowledge distillation from a large LLM into a smaller NMT model can advance the state-of-the-art for low-resource language pairs. Due to compute issues and lack of a curated dataset, I did not replicate their results but sought to utilize their core claim and see if it can work on my setup. The paper's key claims that I have used for this project are:
1. Distillation from a larger teacher into a smaller student is a viable way to reduce parameters and inference cost while maintaining performance, even on a task like NMT, provided the teacher is finetuned well and has demonstrable performance on the specified task.
2. Translating from a different language, say xyz, to English, i.e. [xyz -> eng], is an easier task compared to [eng -> xyz] due to English-centric corpora of most available LLMs, either open-source or commercial.
This project tackles the harder direction as a baseline to see the effects of distillation, since the obverse direction will lead to an obvious jump in results.

### Initial Setup
As any LLM relies on tokenization for understanding and training, choice of language is key given the morphological differences among different languages. Furthermore, as I am someone who does this in my spare time, I have used public open source datasets for this task to avoid the need for collating sentences and curating them. The CCMatrix parallel corpora for multilingual tasks, which was developed using techniques listed at [CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB](https://github.com/facebookresearch/LASER/tree/main/tasks/CCMatrix), and is currently hosted at OPUS, is a popular choice for this task. You can find it at this location:
(https://opus.nlpl.eu/datasets/CCMatrix)


