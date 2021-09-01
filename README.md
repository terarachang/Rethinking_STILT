## Rethinking Why Intermediate-Task Fine-Tuning Works
*RTX3090*

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.7.1](https://img.shields.io/badge/pytorch-1.7.1-green.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **Rethinking Why Intermediate-Task Fine-Tuning Works**<br>
> Ting-Yun Chang and Chi-Jen Lu<br>

> https: https://arxiv.org/pdf/2108.11696.pdf

>
> **Abstract:** *Supplementary Training on Intermediate Labeled-data Tasks (STILT) is a widely applied technique, which first fine-tunes the pretrained language models on an intermediate task before on the target task of interest. While STILT is able to further improve the performance of pretrained language models, it is still unclear why and when it works. Previous research shows that those intermediate tasks involving complex inference, such as commonsense reasoning, work especially well for  RoBERTa-large.
In this paper, we discover that the improvement from an intermediate task could be orthogonal to it containing reasoning or other complex skills --- a simple real-fake discrimination task synthesized by GPT2 can benefit diverse target tasks. We conduct extensive experiments to study the impact of different factors on STILT. These findings suggest rethinking the role of intermediate fine-tuning in the STILT pipeline.*

### Proprecessing
```bash
$ python run_gpt2_synth_dataset.py
```

### Training & Evaluation

#### All Intermediate Tasks
```bash
$ bash run_intermediate_tasks.sh
```

#### All Target Tasks
```bash
$ bash run_target_tasks.sh
```
