
<div align="center">

# Awesome Prompt datasets 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
</div>

<div align="center">

[中文](README_zh.md) | English
</div>

# Contents
- [Awesome Prompt datasets](#awesome-prompt-datasets)
- [Contents](#contents)
- [Introduction](#introduction)
- [Summary](#summary)
- [The template](#the-template)
- [The Prompt Data List](#the-prompt-data-list)
  - [Alpaca -Stanford](#alpaca--stanford)
  - [Instruction in the Wild](#instruction-in-the-wild)
  - [JosephusCheung/GuanacoDataset](#josephuscheungguanacodataset)
  - [Stanford Human Preferences Dataset (SHP)](#stanford-human-preferences-dataset-shp)
    - [Dataset Desig](#dataset-desig)
      - [Domain Selection](#domain-selection)
  - [Hello-SimpleAI/HC3](#hello-simpleaihc3)
  - [Hello-SimpleAI/HC3-Chinese](#hello-simpleaihc3-chinese)
  - [allenai/prosocial-dialog](#allenaiprosocial-dialog)
  - [allenai/natural-instructions](#allenainatural-instructions)
  - [PhoebusSi/Alpaca-CoT](#phoebussialpaca-cot)
  - [nomic-ai/gpt4all](#nomic-aigpt4all)
  - [bigscience/xP3](#bigsciencexp3)
  - [teknium1/GPTeacher](#teknium1gpteacher)
  - [thunlp/UltraChat](#thunlpultrachat)
  - [cascip/ChatAlpaca](#cascipchatalpaca)
  - [YeungNLP/firefly-train-1.1M)](#yeungnlpfirefly-train-11m)
  - [orhonovich/unnatural-instructions](#orhonovichunnatural-instructions)
  - [Instruction-Tuning-with-GPT-4/GPT-4-LLM](#instruction-tuning-with-gpt-4gpt-4-llm)
  - [databrickslabs/dolly](#databrickslabsdolly)
  - [OpenAssistant/oasst1](#openassistantoasst1)
- [Reinforcement Learning from Human Feedback (RLHF) Datasets](#reinforcement-learning-from-human-feedback-rlhf-datasets)
  - [Anthropic/hh-rlhf](#anthropichh-rlhf)
  - [HuggingFaceH4/stack-exchange-preferences](#huggingfaceh4stack-exchange-preferences)
  - [stanfordnlp/SHP](#stanfordnlpshp)
  - [Instruction-Tuning-with-GPT-4/GPT-4-LLM](#instruction-tuning-with-gpt-4gpt-4-llm-1)
  - [Natural Instruction / Super-Natural Instruction](#natural-instruction--super-natural-instruction)
  - [BigScience/P3](#bigsciencep3)
  - [xMTF - BigScience](#xmtf---bigscience)
  - [HH-RLHF - Anthropic](#hh-rlhf---anthropic)
  - [Unnatural Instruction](#unnatural-instruction)
  - [Self-Instruct](#self-instruct)
  - [UnifiedSKG - HKU](#unifiedskg---hku)
  - [Google/Flan Collection](#googleflan-collection)
  - [InstructDial](#instructdial)
  - [ChatGPT Distillation Data](#chatgpt-distillation-data)
  - [Open Instruction Generalist (OIG).](#open-instruction-generalist-oig)
  - [OpenAI WebGPT.](#openai-webgpt)
  - [OpenAI Summarization.](#openai-summarization)
- [Datasets without license information](#datasets-without-license-information)
  - [alespalla/chatbot\_instruction\_prompts](#alespallachatbot_instruction_prompts)
- [Open-source Codebase For Instruction-following LLMs](#open-source-codebase-for-instruction-following-llms)
  - [nichtdax/awesome-totally-open-chatgpt](#nichtdaxawesome-totally-open-chatgpt)
  - [Contributing](#contributing)
  - [License](#license)


# Introduction
"Welcome to 'awesome-prompt-datasets', a comprehensive collection of high-quality open-source instruction tuning datasets to train chat-based LLMs (ChatGPT,LLaMA,Alpaca)。

Instruction Tuning / Reinforcement Learning from Human Feedback (RLHF) Dataset is a key component of instruction-following LLMs such as ChatGPT. This repo is dedicated to providing a comprehensive list of datasets used for instruction tuning in various LLMs, making it easier for researchers and developers to access and utilize these resources.

With 'awesome-prompt-dataset', you can accelerate your research and development in NLP and unlock new opportunities for innovation. Let's explore the possibilities together!"

# Summary

|                      Datasets/Projects                       |              Organization/Author               | Language | Introduction                                                 | Num Rows |
| :----------------------------------------------------------: | :--------------------------------------------: | :------: | ------------------------------------------------------------ | :------: |
| [ Allen AI/Super-Natural Instruction](https://instructions.apps.allenai.org/) |                    Allen AI                    | English  | Contains instruction data of 61 NLP tasks (Natural Instruction) and 1600 NLP tasks (Super-Natural Instruction) |    NA    |
| [PromptSource / P3](https://huggingface.co/datasets/bigscience/P3) |                   BigScience                   | English  | More than 2,000 prompt templates (PromptSource) containing 270 NLP tasks and a P3 dataset with a scale between 100M-1B |    NA    |
| [BigScience/xMTF](https://github.com/bigscience-workshop/xmtf) |                   BigScience                   | English  | Contains 13 NLP tasks and multilingual prompt data in 46 languages |    NA    |
| [Anthropic/HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) |                   Anthropic                    | English  | RLHF dataset designed to train Helpful and Harmless (HH) LLMs |          |
| [Unnatural Instruction](https://github.com/orhonovich/unnatural-instructions) |                   orhonovich                   |          | Use GPT3 to generate 64k instruction prompt data, and get 240k instruction data after rewriting |  240 K   |
|  [Self-Instruct](https://github.com/yizhongw/self-instruct)  |                    yizhongw                    | English  | Using LLMs to generate prompts for instruction-tuning, introducing concepts such as Task pool and Quality filtering |          |
|         [UnifiedSKG - HKU](https://unifiedskg.com/)          |                      HKU                       | English  | Add knowledge grounding to Text-to-Text framework, serialize and embed structured data into prompt |          |
| [Google/Flan Collection](https://github.com/google-research/FLAN/tree/main/flan/v2) |                     Google                     | English  | Merge Flan 2021 data with some open source instruction data (P3, super-natural instruction, etc.) |          |
|                         InstructDial                         |                 prakharguptaz                  | English  | Attempts to fine-tune instructions on a specific task type (dialogue instructions) |          |
|                            Alpaca                            |                    Stanford                    |          | 53k data, very powerful performance (GPT-3.5 level).         |          |
| [webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) |                     Openai                     | English  | In the [WebGPT paper](https://arxiv.org/abs/2112.09332), the authors trained a reward model from human feedback. They used the reward model to train a long form question answering model to align with human preferences. This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total. |  19,578  |
|    [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)    |                  stanfordnlp                   | English  | SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)). |  349 K   |
| [rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets) |                   yitingxie                    | English  |                                                              |  76.3 k  |
| [Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) |                     Dahoas                     | English  | Anthropic's HH dataset reformatted into prompt, chosen, rejected samples. |  112 k   |
| [Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) |    Dahoas/synthetic-instruct-gptj-pairwise     | English  |                                                              |          |
| [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) |                Dahoas/rm-static                | English  | Split of [hh-static](https://huggingface.co/datasets/Dahoas/static-hh) used for training reward models after supervised fine-tuning. |  76.3K   |
| [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) |                 Hello-SimpleAI                 | Chinese  | We propose the first human-ChatGPT comparison corpus, named HC3 dataset. This dataset is introduced in our paper:  Paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597) |          |
| [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |                 Hello-SimpleAI                 | English  |                                                              |  24.3K   |
| [Cohere/miracl-zh-queries-22-12](https://huggingface.co/datasets/Cohere/miracl-zh-queries-22-12) |                     Cohere                     | Chinese  |                                                              |          |
| [wangrui6/Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) |                    wangrui6                    | Chinese  | Zhihu data for training Open Assitant                        |          |
| [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) |                    YeungNLP                    | Chinese  | firefly-train-1.1M：包含了23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万 。此数据应用于应用于项目：[Firefly（流萤）: 中文对话式大语言模型](https://github.com/yangjianxin1/Firefly) |   115M   |
|       [BelleGroup](https://huggingface.co/BelleGroup)        | 项目地址：https://github.com/LianjiaTech/BELLE |          | BELLE Group Dataset : 链家基于ChatGPT用self-instruct生成的中文指令数据集，其中还包括中文数学题数据和多轮对话数据。 |          |

# The template

Append the new project at the end of file
```shell

[{Project-name}/{Dataset-name}]{https://github.com/link/to/project}

- [paper/project link](link)
- [dataset link](link)
- Related work: (if applicable)

Some introductions ...

```

# The Prompt Data List

## [Alpaca -Stanford](https://github.com/tatsu-lab/stanford_alpaca)

- [Paper/Project Link](https://github.com/tatsu-lab/stanford_alpaca)
- [Dataset Link](https://github.com/tatsu-lab/stanford_alpaca)
- Data generation model: text-davinci-003
- Cost: $600

The Alpaca of the Stanford release is a fine-tuning model for instruct-tuning based on the Meta Ai LLaMA model.

Alpaca automatically generated 52k instruction data using GPT-3.5 and used it to fine-tune the LLaMA model. Experimental results show that it can reach or even exceed the performance of GPT-3.5 on some tasks.

## [Instruction in the Wild](https://github.com/XueFuzhao/InstructionWild)

- [Paper/Project Link](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)
- [Dataset Link](https://github.com/XueFuzhao/InstructionWild)
- Data generation model: text-davinci-003
  

Instruction Tuning is a key component of ChatGPT. OpenAI used their user-based Instruction dataset, but unfortunately, this dataset is not open-sourced. Self-Instruct released a small instruction dataset including 175 instructions written by human labors. Standford Alpaca Team generated 52K instructions by text-davinci-003 model based on the the 175 seed instructions above.

This project targets on a larger and more diverse instruction dataset. To this end, we collected 429 instructions from ChatGPT usage screenshots and released both English and Chinese versions. We found these instructions are very diverse even if the scale is still small. We follow Alpaca to generate 52K instructions and their responses. All data can be found in data dir.

Note: This is an ongoing project. We are still collecting and improving our data. We release this dataset as early as possible to speedup our LLM research. We will also release a whitepaper soon.

## [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

- Data generation model: text-davinci-003
- Cost: $6000

52K instruction data generated from modified self-instruct pipeline with human written 429 seed task.


##  [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)

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

### Dataset Desig

#### Domain Selection

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


## [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

- Summary:The the first human-ChatGPT comparison corpus (English Version), named HC3 dataset
- Data generation model: `gpt-3.5`, `human generated`
- paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
- Cost: N/A

## [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

- Summary:The the first human-ChatGPT comparison corpus (Chinese Version), named HC3 dataset
- Data generation model: `gpt-3.5`, `human generated`
- paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
- Cost: N/A


## [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)

- Summary: ProsocialDialog is the first large-scale multi-turn English dialogue dataset to teach conversational agents to respond to problematic content following social norms.
- Data generation model: `gpt-3.5`, `human generated`
- paper: [ProsocialDialog: A Prosocial Backbone for Conversational Agents](https://arxiv.org/abs/2205.12688)
- Cost: N/A

## [allenai/natural-instructions](https://github.com/allenai/natural-instructions)

- Summary: A community effort to create a large collection of `1,616 diverse NLP tasks` and their natural language definitions/instructions.
- Data generation model: `Human generated`
- paper: [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705)
- Cost: N/A


## [PhoebusSi/Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)

- Summary: A datset for Chain-of-Thoughts reasoning based on LLaMA and Alpaca. Note: Their repository will continuously collect various instruction tuning datasets. [Github Repo](https://github.com/PhoebusSi/Alpaca-CoT)
- paper: N/A
- Cost: N/A

## [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)

- Summary: gpt4all leverages three publicly available datasets: 1.[laion/OIG](https://huggingface.co/datasets/laion/OIG), 2.[pacovaldez/stackoverflow-questions](https://huggingface.co/datasets/pacovaldez/stackoverflow-questions) 3. subset of [bigscience/bloomz-p3](https://huggingface.co/bigscience/bloomz-p3)
- Data generation model: N/A
- paper: [GPT4All: Training an Assistant-style Chatbot with Large Scale Data Distillation from GPT-3.5-Turbo](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)
- Cost: $500


## [bigscience/xP3](https://huggingface.co/datasets/bigscience/xP3)

- Summary: [Prompt-resource] xP3 (Crosslingual Public Pool of Prompts) is a collection of prompts & datasets across 46 of languages & 16 NLP tasks.
- Data generation model: N/A
- paper: [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
- Cost: N/A



## [teknium1/GPTeacher](https://github.com/teknium1/GPTeacher)

- Summary: A collection of modular datasets generated by GPT-4, General-Instruct - Roleplay-Instruct - Code-Instruct - and Toolformer
- Data generation model: `GPT-4`
- paper: N/A
- Cost: N/A



## [thunlp/UltraChat](https://github.com/thunlp/UltraChat)

- Summary: UltraChat aims to construct an open-source, large-scale, and multi-round dialogue data. The first part of UltraChat (i.e., the Questions about the World sector) is released, which contains 280k diverse and informative dialogues. More dialogues about writing and creation, assistance on existing materials are to come.
- Data generation model: `GPT-3.5-turbo`
- paper: N/A
- Cost: N/A

## [cascip/ChatAlpaca](https://github.com/cascip/ChatAlpaca)

- Summary: Based on the Stanford Alpaca data, ChatAlpaca extends the data to multi-turn instructions and their corresponding responses. More data (20k) and the Chinese translated version are to come.
- Data generation model: `GPT-3.5-turbo`
- paper: N/A
- Cost: N/A
- Related: [(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)

## [YeungNLP/firefly-train-1.1M)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- Summary: Chinese datasets of 23 tasks combined with human-written instruction templates. 
- Data generation model: N/A
- paper: N/A
- Cost: N/A

## [orhonovich/unnatural-instructions](https://github.com/orhonovich/unnatural-instructions)
- Summary: 64K examples by prompting a language model with three seed examples of instructions and eliciting a fourth. Then the set is expanded to 240K by prompting the model to rephrase each instruction.
- Data generation model: `text-davinci-002`
- paper: [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)
- Cost: N/A

## [Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- Summary: 52K instruction-following data generated by GPT-4 with the original Alpaca prompts & Alpaca prompts translated into Chinese by ChatGPT + 9K instruction-following data generated by GPT-4 with prompts in Unnatural Instruction.
- Data generation model: `GPT-4`
- paper: [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)
- Cost: N/A
- Related: 
    -[(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)
    -[(orhonovich/unnatural-instructions)|240K|EN|MT|MIX](https://github.com/orhonovich/unnatural-instructions)

## [databrickslabs/dolly](https://github.com/databrickslabs/dolly/tree/master/data)
- Summary: This datset was generated by thousands of Databricks employees in several of the behavioral categories outlined in the InstructGPT paper, including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization.
- Data generation model: N/A
- paper: [Free Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- Cost: N/A

## [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- Summary: OpenAssistant Conversations (OASST1), a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages distributed across 66,497 conversation trees, in 35 different languages, annotated with 461,292 quality ratings. 
- Data generation model: N/A
- paper: [OpenAssistant Conversations - Democratizing Large Language Model Alignment](https://drive.google.com/file/d/10iR5hKwFqAKhL3umx8muOWSRm7hs5FqX/view)
- Cost: N/A

## BELLE/data/1.5M

- 下载地址: [https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)
- 数据量: 1.5M
- 生成方式: self-instruct，使用了中文种子任务，以及openai的text-davinci-003接口
- 涉及任务: 包含175个种子任务，[https://github.com/LianjiaTech/BELLE/blob/main/data/1.5M/zh_seed_tasks.json](https://github.com/LianjiaTech/BELLE/blob/main/data/1.5M/zh_seed_tasks.json)
- 数据示例: [https://huggingface.co/datasets](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

## alpaca_chinese_dataset

- 下载地址: [https://github.com/hikariming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset)
- 数据量: 52k
- 生成方式: 借助chatgpt对原始的[stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)做机器翻译，并加入人工校验来保证质量
- 涉及任务: 与原始的[stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)一致，可以在原项目的[seed_task.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl)中查到全部任务

## Med-ChatGLM/data

- 下载地址: [https://github.com/SCIR-HI/Med-ChatGLM](https://github.com/SCIR-HI/Med-ChatGLM)
- 数据量: 7k
- 生成方式: 利用GPT3.5接口围绕医学知识库构建问答数据，并设置了多种Prompt形式来充分利用知识
- 涉及任务: 医学领域相关的问答，包含并发症，高危因素，组织学检查，临床症状，药物治疗，辅助治疗

## pCLUE

- 下载地址: [https://github.com/CLUEbenchmark/pCLUE](https://github.com/CLUEbenchmark/pCLUE)
- 数据量: 1.2M
- 生成方式: 通过原有的NLP任务数据集，结合特定的[prompt](https://www.zhihu.com/search?q=prompt&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"624084039"})模板生成
- 涉及任务: 包含9个NLP数据集，涉及的NLP任务有[文本分类](https://www.zhihu.com/search?q=文本分类&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"624084039"})/自然语言推理/语义匹配/[指代消解](https://www.zhihu.com/search?q=指代消解&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"624084039"})/关键词识别/阅读理解

### COIG

- 下载地址: [https://huggingface.co/datasets/BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)

- 数据量: 

- - Translated Instructions (67,798)
  - Exam Instructions (63,532)
  - Human Value Alignment Instructions (34,471)
  - Counterfactural Correction Multi-round Chat (13,653)
  - Leetcode Instructions (11,737)

- 生成方式: 融合了多个领域的数据，具体可以参考论文[Chinese Open Instruction Generalist: A Preliminary Release](https://arxiv.org/abs/2304.07987)

https://github.com/FreedomIntelligence/InstructionZoo

https://github.com/lightaime/camel

# Reinforcement Learning from Human Feedback (RLHF) Datasets

## [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- Summary: This RLHF dataset is an iterated 'online' dataset that includes data from 52B language models. It contains 22k helpfulness comparisons and no red-teaming data. 
- Data generation model: `Anthropic RL-CAI 52B`
- paper: [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- Cost: N/A
- Related: 
    -[(Hello-SimpleAI/HC3)|24K|EN|MT|MIX](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
    -[(Hello-SimpleAI/HC3-Chinese)|13K|CN|MT|MIX](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

## [HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)

- Summary: This dataset contains questions and answers from the Stack Overflow Data Dump for the purpose of preference model training.
- Data generation model: N/A
- paper: [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
- Cost: N/A

## [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)

- Summary: Each example is a Reddit post with a question/instruction and a pair of top-level comments for that post, where one comment is more preferred by Reddit users (collectively).
- Data generation model: N/A
- paper: N/A
- Cost: N/A

## [Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

- Summary: Ranked responses (Note: Data is evaluated by `GPT-4` model NOT human) of Alpaca prompts from three models (GPT-4, GPT-3.5 and OPT-IML) by asking GPT-4 to rate the quality. Author believes "GPT-4 is capable of identifying and fixing its own mistakes, and accurately judging the quality of responses" 
- Data generation model: `GPT-4`
- paper: [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)
- Cost: N/A
- Related: 
    -[(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)


## Natural Instruction / Super-Natural Instruction

- [Paper/Project link](https://aclanthology.org/2022.acl-long.244.pdf)
- [Dataset link](https://instructions.apps.allenai.org/)

Allen AI is the first organization to try Instruction as a prompt and fine-tune LLMs. In the Natural Instruction paper, you can basically understand the labeling ideas of the instruction.

In its proposed dataset, 61 and different NLP tasks are included.

Super-Natural Instruction is a super-intensive version of Natural Instruction, which contains more than 1,600 different NLP tasks, and there are more than 76 different types of NLP tasks (such as: classification, extraction, sequence labeling).

## [BigScience/P3](https://huggingface.co/datasets/bigscience/P3)

- [Paper/Project Link](https://github.com/bigscience-workshop/promptsource)
- [Dataset Link](https://huggingface.co/datasets/bigscience/P3)

BigScience is jointly organized by Hugging Face and French CNRS, IDRIS, GENCI, etc. It is one of the largest open source LLMs organizations.

BigScience developed the PromptSource project at the end of 2021, and open sourced a series of toolkits to help researchers build prompts based on existing NLP tasks. So far, the PromptSource project contains more than 2000 prompt templates for 270 NLP tasks.

On this basis, BigScience constructed the P3 dataset. You can find P3 data on Hugging Face Hub, and the data size of P3 is between 100M-1B.

## xMTF - BigScience

- [Project Link](https://arxiv.org/pdf/2211.01786.pdf)
- [Dataset Link](https://github.com/bigscience-workshop/xmtf)

Based on the English prompt, BigScience extends its prompt to multiple non-English languages.

The project contains 13 NLP tasks and is available in 46 different languages. The corresponding prompt contains an indeterminate number of languages.

After fine-tuning on the basis of multilingual, both BLOOM and T0 have realized the ideal multilingual ability.

## HH-RLHF - Anthropic

- [Paper/Project Link](https://arxiv.org/pdf/2204.05862.pdf)
- [Dataset Link](https://huggingface.co/datasets/Anthropic/hh-rlhf)

Claud under Anthropic is one of the main competitors of ChatGPT.

Anthropic has open-sourced the RLHF dataset it uses in its own product line.

The original intention of the HH-RLHF project is to train Helpful and Harmless (HH) LLMs. Therefore, in addition to the quality of the project's responses, whether it is harmful information is also reflected in its human feedback.

The paper records how to use the behavior of the RLHF data Align model to human values, and records the construction method and standards of the data set.

## [Unnatural Instruction](https://github.com/orhonovich/unnatural-instructions)

- [Paper/Project Link](https://arxiv.org/pdf/2212.09689.pdf)
- [Dataset Link](https://github.com/orhonovich/unnatural-instructions)

Using LLMs to independently generate instruction data is an active direction in the field of instruction-tuning.

Unnatural Instruction uses GPT3 (text-davinci-002) to generate 64k instruction prompt data. And use the same model to rewrite the 64k prompt, and finally get 240k instruction data.

The paper shows that the prompts generated by LLMs in Instruct-Tuning show good results, even surpassing models such as T0 that are fine-tuned on P3 and other data.

## [Self-Instruct](https://github.com/yizhongw/self-instruct)

- [Paper/Project Link](https://arxiv.org/pdf/2212.10560.pdf)
- [Dataset Link](https://github.com/yizhongw/self-instruct)

Self-Instruct is also the idea of using LLMs to generate prompts for instruction-tuning. However, a more fine-grained generation process is used.

Concepts such as Task pool and Quality filtering were introduced to partially alleviate the noise problem of self-intrauct type data.

## [UnifiedSKG - HKU](https://unifiedskg.com/)

- [Paper/Project Link](https://arxiv.org/pdf/2201.05966.pdf)

- [DataSet Link](https://unifiedskg.com/)

UnifiedSKG has added knowledge grounding in the Text-to-Text framework, that is, in the prompt-output framework, it has added structured data for assistance.

As an example, some NLP tasks rely heavily on structured knowledge bases/databases. The idea of UnifiedSKG is to serialize the required database and embed it into the prompt. As shown below.

UnifiedSKG represents a direction in the field of LLMs that attempts to use structured knowledge to enhance performance.

## [Google/Flan Collection](https://github.com/google-research/FLAN/tree/main/flan/v2)

- [Paper/Project Link](https://arxiv.org/pdf/2301.13688.pdf)
- [Dataset Link](https://github.com/google-research/FLAN/tree/main/flan/v2)

In this project, Google merged its own Flan 2021 data with some open source instruction data (P3, super-natural instruction, etc.).

In Flan Collection's paper, Google also summarizes some key points in Flan series model training/reasoning, which may have good reference value.

The Flan Collection compiles datasets from Flan 2021, P3, Super-Natural Instructions, along with dozens more datasets into one place, formats them into a mix of zero-shot, few-shot and chain-of-thought templates

- 
## InstructDial

- [Paper/Project Link](https://arxiv.org/pdf/2205.12673.pdf)
- [Dataset Link](https://github.com/prakharguptaz/Instructdial/tree/main/datasets)

InstructDial is an attempt to fine-tune instructions on a specific task type. Experimental results show that after fine-tuning on dialogue instruction data, the model performs better on dialogue tasks than on very large-scale task sets.


## ChatGPT Distillation Data
Public User-Shared Dialogues with ChatGPT (ShareGPT) Around 60K dialogues shared by users on ShareGPT were collected using public APIs. To maintain data quality, we deduplicated on the user-query level and removed any non-English conversations. This leaves approximately 30K examples.

Human ChatGPT Comparison Corpus (HC3) We use both the human and ChatGPT responses from the [HC3 english dataset](https://arxiv.org/abs/2301.07597), which contains around 60K human answers and 27K ChatGPT answers for around 24K questions, resulting in a total number of around 87K question-answer examples.


## Open Instruction Generalist (OIG). 
- [Paper/Project Link](https://arxiv.org/abs/2106.03300)
- [Dataset Link](https://laion.ai/blog/oig-dataset/)

We use a manually-selected subset of components from the Open [Instruction Generalist dataset](https://laion.ai/blog/oig-dataset/) curated by LAION. Specifically, we use the grade-school-math-instructions, the poetry-to-songs, and the plot-screenplay-books-dialogue datasets. This results in a total of around 30k examples.


## OpenAI WebGPT. 
- [Paper/Project Link](https://arxiv.org/abs/2106.03300)
- [Dataset Link](https://huggingface.co/datasets/openai/webgpt_comparisons)

In the [WebGPT paper](https://arxiv.org/abs/2112.09332), the authors trained a reward model from human feedback. They used the reward model to train a long form question answering model to align with human preferences. This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total.

Each example in the dataset contains a pair of model answers for a question, and the associated metadata. Each answer has a preference score from humans that can be used to determine which of the two answers are better. 

## OpenAI Summarization. 
- [Paper/Project Link](https://arxiv.org/abs/2106.03300)
- [Dataset Link](https://huggingface.co/datasets/openai/summarization)

The OpenAI summarization dataset contains ~93K examples, each example consists of feedback from humans regarding the summarizations generated by a model. Human evaluators chose the superior summary from two options.





# Datasets without license information 

 ## [alespalla/chatbot_instruction_prompts](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts)

 - Summary: A compilation of `tatsu-lab/alpaca` ,`Dahoas/instruct-human-assistant-prompt` ,`allenai/prosocial-dialog`
 - Data generation model: N/A
 - paper: N/A
 - Cost: N/A

# Open-source Codebase For Instruction-following LLMs

## [nichtdax/awesome-totally-open-chatgpt](https://github.com/nichtdax/awesome-totally-open-chatgpt)
- Summary: Alternatives are projects featuring different instruct finetuned language models for chat. 

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License

`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.