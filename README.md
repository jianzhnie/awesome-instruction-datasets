
<div align="center">

# Awesome Prompt datasets 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
</div>

<div align="center">

[中文](README_zh.md) | English
</div>

## Contents
- [Awesome Prompt datasets](#awesome-prompt-datasets)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Summary](#summary)
  - [The template](#the-template)
  - [The Prompt Data List](#the-prompt-data-list)
    - [Natural Instruction / Super-Natural Instruction](#natural-instruction--super-natural-instruction)
    - [PromptSource / P3](#promptsource--p3)
    - [xMTF - BigScience](#xmtf---bigscience)
    - [HH-RLHF - Anthropic](#hh-rlhf---anthropic)
    - [Unnatural Instruction](#unnatural-instruction)
    - [Self-Instruct](#self-instruct)
    - [UnifiedSKG - HKU](#unifiedskg---hku)
    - [Flan Collection - Google](#flan-collection---google)
    - [InstructDial](#instructdial)
    - [Alpaca -Stanford](#alpaca--stanford)
    - [Instruction in the Wild](#instruction-in-the-wild)
    - [ChatGPT Distillation Data](#chatgpt-distillation-data)
    - [Open Instruction Generalist (OIG).](#open-instruction-generalist-oig)
    - [OpenAI WebGPT.](#openai-webgpt)
    - [OpenAI Summarization.](#openai-summarization)
  - [Contributing](#contributing)
  - [License](#license)


## Introduction
"Welcome to 'awesome-prompt-datasets', a comprehensive collection of high-quality prompts for natural language processing tasks. With 'awesome-prompt-dataset', you can accelerate your research and development in NLP and unlock new opportunities for innovation. Let's explore the possibilities together!"

## Summary

|                      Datasets/Projects                       |           Organization/Author           | Language | Introduction                                                 | Num Rows |
| :----------------------------------------------------------: | :-------------------------------------: | :------: | ------------------------------------------------------------ | :------: |
|       Natural Instruction / Super-Natural Instruction        |                Allen AI                 |          | Contains instruction data of 61 NLP tasks (Natural Instruction) and 1600 NLP tasks (Super-Natural Instruction) |          |
|                      PromptSource / P3                       |               BigScience                |          | More than 2,000 prompt templates (PromptSource) containing 270 NLP tasks and a P3 dataset with a scale between 100M-1B |          |
|                             xMTF                             |               BigScience                |          | Contains 13 NLP tasks and multilingual prompt data in 46 languages |          |
|                           HH-RLHF                            |                Anthropic                |          | RLHF dataset designed to train Helpful and Harmless (HH) LLMs |          |
|                    Unnatural Instruction                     |               orhonovich                |          | Use GPT3 to generate 64k instruction prompt data, and get 240k instruction data after rewriting |          |
|                        Self-Instruct                         |                yizhongw                 |          | Using LLMs to generate prompts for instruction-tuning, introducing concepts such as Task pool and Quality filtering |          |
|                          UnifiedSKG                          |                   HKU                   |          | Add knowledge grounding to Text-to-Text framework, serialize and embed structured data into prompt |          |
|                       Flan Collection                        |                 Google                  |          | Merge Flan 2021 data with some open source instruction data (P3, super-natural instruction, etc.) |          |
|                         InstructDial                         |              prakharguptaz              |          | Attempts to fine-tune instructions on a specific task type (dialogue instructions) |          |
|                            Alpaca                            |                Stanford                 |          | 53k data, very powerful performance (GPT-3.5 level).         |          |
|                                                              |                                         |          |                                                              |          |
| [webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) |                 Openai                  | English  | In the [WebGPT paper](https://arxiv.org/abs/2112.09332), the authors trained a reward model from human feedback. They used the reward model to train a long form question answering model to align with human preferences. This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total. |  19,578  |
|    [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)    |               stanfordnlp               | English  | SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)). |  349 K   |
| [rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets) |                yitingxie                | English  |                                                              |  76.3 k  |
| [Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) |                 Dahoas                  | English  | Anthropic's HH dataset reformatted into prompt, chosen, rejected samples. |  112 k   |
| [Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | Dahoas/synthetic-instruct-gptj-pairwise | English  |                                                              |          |
| [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) |            Dahoas/rm-static             | English  | Split of [hh-static](https://huggingface.co/datasets/Dahoas/static-hh) used for training reward models after supervised fine-tuning. |  76.3K   |
| [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) |             Hello-SimpleAI              | Chinese  | We propose the first human-ChatGPT comparison corpus, named HC3 dataset. This dataset is introduced in our paper:  Paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597) |          |
| [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |             Hello-SimpleAI              | English  |                                                              |  24.3K   |
| [Cohere/miracl-zh-queries-22-12](https://huggingface.co/datasets/Cohere/miracl-zh-queries-22-12) |                 Cohere                  | Chinese  |                                                              |          |
| [wangrui6/Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) |                wangrui6                 | Chinese  | Zhihu data for training Open Assitant                        |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |
|                                                              |                                         |          |                                                              |          |

## The template

Append the new project at the end of file
```shell

[{Project-name}/{Dataset-name}]{https://github.com/link/to/project}

- [paper/project link](link)
- [dataset link](link)

Some introductions ...
```

## The Prompt Data List

###  Stanford Human Preferences Dataset (SHP)

- [DataLinks](https://huggingface.co/datasets/stanfordnlp/SHP)

SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)).

Each example is a Reddit post with a question/instruction and a pair of top-level comments for that post, where one comment is more preferred by Reddit users (collectively). SHP exploits the fact that if comment A was written after comment B but has a higher score nonetheless, then A is ostensibly more preferred to B. If A had been written before B, then we could not conclude this, since its higher score could have been the result of more visibility. We chose data where the preference label is intended to reflect which response is more helpful rather than which is less harmful, the latter being the focus of much past work.

How is SHP different from [Anthropic's HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)? Most notably, all the data in SHP is naturally occurring and human-written, whereas the responses in HH-RLHF are machine-written, giving us two very different distributions that can complement each other.

| Dataset | Size | Input                                       | Label                       | Domains       | Data Format                                   | Length                |
| ------- | ---- | ------------------------------------------- | --------------------------- | ------------- | --------------------------------------------- | --------------------- |
| SHP     | 385K | Naturally occurring human-written responses | Collective Human Preference | 18 (labelled) | Question/Instruction + Response (Single-turn) | up to 10.1K T5 tokens |
| HH-RLHF | 91K  | Dialogue with LLM                           | Individual Human Preference | not labelled  | Live Chat (Multi-turn)                        | up to 1.5K T5 tokens  |

How is SHP different from other datasets that have scraped Reddit, like [ELI5](https://huggingface.co/datasets/eli5#source-data)? SHP uses the timestamp information to infer preferences, while ELI5 only provides comments and scores -- the latter are not enough to infer preferences since comments made earlier tend to get higher scores from more visibility. It also contains data from more domains:

| Dataset | Size | Comments + Scores | Preferences | Number of Domains |
| ------- | ---- | ----------------- | ----------- | ----------------- |
| SHP     | 385K | Yes               | Yes         | 18                |
| ELI5    | 270K | Yes               | No          | 3                 |

#### Dataset Desig

##### Domain Selection

The data is sourced from Reddit, which is a public forum organized into topic-specific fora called subreddits. For example, the `askculinary` subreddit is where users ask cooking-related questions and are answered by other users.

SHP contains a train, validation, and test split for comments scraped from 18 different subreddits. We chose subreddits based on:

1. whether they were well-known (subscriber count >= 100K)
2. whether posts were expected to pose a question or instruction
3. whether responses were valued based on how helpful they were
4. whether comments had to be rooted in some objectivity, instead of being entirely about personal experiences (e.g., `askscience` vs. `AskAmericans`)

The train/validation/test splits were created by splitting the post IDs of a subreddit in 90%/5%/5% proportions respectively, so that no post would appear in multiple splits. Since different posts have different numbers of comments, the number of preferences in each split is not exactly 90%/5%/5%:

| subreddit         | train  | validation | test  | total  |
| ----------------- | ------ | ---------- | ----- | ------ |
| askacademia       | 31450  | 2095       | 1708  | 35253  |
| askanthropology   | 3910   | 203        | 268   | 4381   |
| askbaking         | 44007  | 2096       | 1544  | 47647  |
| askcarguys        | 3227   | 159        | 117   | 3503   |
| askculinary       | 45710  | 2094       | 2563  | 50367  |
| askdocs           | 6449   | 315        | 455   | 7219   |
| askengineers      | 57096  | 3154       | 2638  | 62888  |
| askhistorians     | 3264   | 113        | 164   | 3541   |
| askhr             | 8295   | 641        | 395   | 9331   |
| askphilosophy     | 10307  | 608        | 677   | 11592  |
| askphysics        | 7364   | 409        | 587   | 8360   |
| askscience        | 13316  | 899        | 977   | 15192  |
| asksciencefiction | 29382  | 1576       | 1987  | 32945  |
| asksocialscience  | 2706   | 147        | 188   | 3041   |
| askvet            | 3300   | 170        | 224   | 3694   |
| changemyview      | 38173  | 1637       | 1836  | 41646  |
| explainlikeimfive | 19592  | 1014       | 1070  | 21676  |
| legaladvice       | 21170  | 1106       | 1011  | 23287  |
| ALL               | 348718 | 18436      | 18409 | 385563 |

### Natural Instruction / Super-Natural Instruction

- [Paper/Project link](https://aclanthology.org/2022.acl-long.244.pdf)
- [Dataset link](https://instructions.apps.allenai.org/)

Allen AI is the first organization to try Instruction as a prompt and fine-tune LLMs. In the Natural Instruction paper, you can basically understand the labeling ideas of the instruction.

In its proposed dataset, 61 and different NLP tasks are included.

Super-Natural Instruction is a super-intensive version of Natural Instruction, which contains more than 1,600 different NLP tasks, and there are more than 76 different types of NLP tasks (such as: classification, extraction, sequence labeling).

### PromptSource / P3

- [Paper/Project Link](https://github.com/bigscience-workshop/promptsource)
- [Dataset Link](https://huggingface.co/datasets/bigscience/P3)

BigScience is jointly organized by Hugging Face and French CNRS, IDRIS, GENCI, etc. It is one of the largest open source LLMs organizations.

BigScience developed the PromptSource project at the end of 2021, and open sourced a series of toolkits to help researchers build prompts based on existing NLP tasks. So far, the PromptSource project contains more than 2000 prompt templates for 270 NLP tasks.

On this basis, BigScience constructed the P3 dataset. You can find P3 data on Hugging Face Hub, and the data size of P3 is between 100M-1B.

### xMTF - BigScience

- [Project Link](https://arxiv.org/pdf/2211.01786.pdf)
- [Dataset Link](https://github.com/bigscience-workshop/xmtf)

Based on the English prompt, BigScience extends its prompt to multiple non-English languages.

The project contains 13 NLP tasks and is available in 46 different languages. The corresponding prompt contains an indeterminate number of languages.

After fine-tuning on the basis of multilingual, both BLOOM and T0 have realized the ideal multilingual ability.

### HH-RLHF - Anthropic

- [Paper/Project Link](https://arxiv.org/pdf/2204.05862.pdf)
- [Dataset Link](https://huggingface.co/datasets/Anthropic/hh-rlhf)

Claud under Anthropic is one of the main competitors of ChatGPT.

Anthropic has open-sourced the RLHF dataset it uses in its own product line.

The original intention of the HH-RLHF project is to train Helpful and Harmless (HH) LLMs. Therefore, in addition to the quality of the project's responses, whether it is harmful information is also reflected in its human feedback.

The paper records how to use the behavior of the RLHF data Align model to human values, and records the construction method and standards of the data set.

### Unnatural Instruction

- [Paper/Project Link](https://arxiv.org/pdf/2212.09689.pdf)
- [Dataset Link](https://github.com/orhonovich/unnatural-instructions)

Using LLMs to independently generate instruction data is an active direction in the field of instruction-tuning.

Unnatural Instruction uses GPT3 (text-davinci-002) to generate 64k instruction prompt data. And use the same model to rewrite the 64k prompt, and finally get 240k instruction data.

The paper shows that the prompts generated by LLMs in Instruct-Tuning show good results, even surpassing models such as T0 that are fine-tuned on P3 and other data.

### Self-Instruct

- [Paper/Project Link](https://arxiv.org/pdf/2212.10560.pdf)
- [Dataset Link](https://github.com/yizhongw/self-instruct)

Self-Instruct is also the idea of using LLMs to generate prompts for instruction-tuning. However, a more fine-grained generation process is used.

Concepts such as Task pool and Quality filtering were introduced to partially alleviate the noise problem of self-intrauct type data.

### UnifiedSKG - HKU

- [Paper/Project Link](https://arxiv.org/pdf/2201.05966.pdf)

- [DataSet Link](https://unifiedskg.com/)

UnifiedSKG has added knowledge grounding in the Text-to-Text framework, that is, in the prompt-output framework, it has added structured data for assistance.

As an example, some NLP tasks rely heavily on structured knowledge bases/databases. The idea of UnifiedSKG is to serialize the required database and embed it into the prompt. As shown below.

UnifiedSKG represents a direction in the field of LLMs that attempts to use structured knowledge to enhance performance.

### Flan Collection - Google

- [Paper/Project Link](https://arxiv.org/pdf/2301.13688.pdf)
- [Dataset Link](https://github.com/google-research/FLAN/tree/main/flan/v2)

In this project, Google merged its own Flan 2021 data with some open source instruction data (P3, super-natural instruction, etc.).

In Flan Collection's paper, Google also summarizes some key points in Flan series model training/reasoning, which may have good reference value.

### InstructDial

- [Paper/Project Link](https://arxiv.org/pdf/2205.12673.pdf)
- [Dataset Link](https://github.com/prakharguptaz/Instructdial/tree/main/datasets)

InstructDial is an attempt to fine-tune instructions on a specific task type. Experimental results show that after fine-tuning on dialogue instruction data, the model performs better on dialogue tasks than on very large-scale task sets.

### Alpaca -Stanford

- [Paper/Project Link](https://github.com/tatsu-lab/stanford_alpaca)
- [Dataset Link](https://github.com/tatsu-lab/stanford_alpaca)

The Alpaca of the Stanford release is a fine-tuning model for instruct-tuning based on the Meta Ai LLaMA model.

Alpaca automatically generated 52k instruction data using GPT-3.5 and used it to fine-tune the LLaMA model. Experimental results show that it can reach or even exceed the performance of GPT-3.5 on some tasks.


### Instruction in the Wild
- [Paper/Project Link](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
- [Dataset Link](https://github.com/XueFuzhao/InstructionWild)
  

Instruction Tuning is a key component of ChatGPT. OpenAI used their user-based Instruction dataset, but unfortunately, this dataset is not open-sourced. Self-Instruct released a small instruction dataset including 175 instructions written by human labors. Standford Alpaca Team generated 52K instructions by text-davinci-003 model based on the the 175 seed instructions above.

This project targets on a larger and more diverse instruction dataset. To this end, we collected 429 instructions from ChatGPT usage screenshots and released both English and Chinese versions. We found these instructions are very diverse even if the scale is still small. We follow Alpaca to generate 52K instructions and their responses. All data can be found in data dir.

Note: This is an ongoing project. We are still collecting and improving our data. We release this dataset as early as possible to speedup our LLM research. We will also release a whitepaper soon.


### ChatGPT Distillation Data
Public User-Shared Dialogues with ChatGPT (ShareGPT) Around 60K dialogues shared by users on ShareGPT were collected using public APIs. To maintain data quality, we deduplicated on the user-query level and removed any non-English conversations. This leaves approximately 30K examples.

Human ChatGPT Comparison Corpus (HC3) We use both the human and ChatGPT responses from the [HC3 english dataset](https://arxiv.org/abs/2301.07597), which contains around 60K human answers and 27K ChatGPT answers for around 24K questions, resulting in a total number of around 87K question-answer examples.


### Open Instruction Generalist (OIG). 
- [Paper/Project Link](https://arxiv.org/abs/2106.03300)
- [Dataset Link](https://laion.ai/blog/oig-dataset/)

We use a manually-selected subset of components from the Open [Instruction Generalist dataset](https://laion.ai/blog/oig-dataset/) curated by LAION. Specifically, we use the grade-school-math-instructions, the poetry-to-songs, and the plot-screenplay-books-dialogue datasets. This results in a total of around 30k examples.


### OpenAI WebGPT. 
- [Paper/Project Link](https://arxiv.org/abs/2106.03300)
- [Dataset Link](https://huggingface.co/datasets/openai/webgpt_comparisons)

In the [WebGPT paper](https://arxiv.org/abs/2112.09332), the authors trained a reward model from human feedback. They used the reward model to train a long form question answering model to align with human preferences. This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total.

Each example in the dataset contains a pair of model answers for a question, and the associated metadata. Each answer has a preference score from humans that can be used to determine which of the two answers are better. 

### OpenAI Summarization. 
- [Paper/Project Link](https://arxiv.org/abs/2106.03300)
- [Dataset Link](https://huggingface.co/datasets/openai/summarization)

The OpenAI summarization dataset contains ~93K examples, each example consists of feedback from humans regarding the summarizations generated by a model. Human evaluators chose the superior summary from two options.


## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License

`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.