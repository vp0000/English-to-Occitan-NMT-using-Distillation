# Distillation on open-source LLMs for English to Occitan NMT

Hi, everyone!
This repo hosts a pipeline for translating English sentences to Occitan using a Mistral-7B-v0.3 model distilled into a TinyLlama-1.1B-Chat-v0.1 model.
The model fine-tuning and knowledge distillation utilises the CCMatrix English-Occitan parallel corpus of 1.8M sentences, and the performance has been evaluated using a
held-out set of FLORES-200, alongside a dedicated Marian MT baseline for comparison.

## Motivation

This personal project was primarily an attempt to understand knowledge distillation in LLMs and their applications for machine translation. The initial idea came about after
reading this paper by Maxim Enis and Mark Hopkins:
@misc{enis2024llmnmtadvancinglowresource,
      title={From LLM to NMT: Advancing Low-Resource Machine Translation with Claude}, 
      author={Maxim Enis and Mark Hopkins},
      year={2024},
      eprint={2404.13813},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.13813}, 
}



