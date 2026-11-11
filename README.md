
<div align="center">

# Awesome Instruction Datasets
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--06--17-blue)
</div>

<div align="center">

[中文](README_zh.md) | English
</div>

# Contents

- [Introduction](#introduction)
- [What's New (2023–2026)](#whats-new-20232026)
- [Quick Index by Category](#quick-index-by-category)
- [How to use this list](#how-to-use-this-list)
- [SFT Datasets](#sft-datasets)
  - [SFT Statistics](#sft-statistics)
    - [General SFT](#general-sft)
    - [Code & Math](#code--math)
    - [Multilingual & Chinese](#multilingual--chinese)
    - [Agent & Tool Use](#agent--tool-use)
  - [SFT Dataset Details](#sft-dataset-details)
- [RLHF / Preference Datasets](#rlhf--preference-datasets)
  - [RLHF Statistics](#rlhf-statistics)
  - [RLHF Dataset Details](#rlhf-dataset-details)
- [DPO Datasets](#dpo-datasets)
  - [DPO Statistics](#dpo-statistics)
  - [DPO Dataset Details](#dpo-dataset-details)
- [ModelScope Chinese Datasets](#modelscope-chinese-datasets)
- [Datasets without license information](#datasets-without-license-information)
- [Contributing](#contributing)
- [License](#license)
- [Reference](#reference)


<a id="introduction"></a>
# Introduction

"Welcome to 'awesome-prompt-datasets', a comprehensive collection of high-quality open-source instruction tuning datasets to train chat-based LLMs (ChatGPT, LLaMA, Alpaca).

Instruction Tuning / Reinforcement Learning from Human Feedback (RLHF) Dataset is a key component of instruction-following LLMs such as ChatGPT. This repo is dedicated to providing a comprehensive list of datasets used for instruction tuning in various LLMs, making it easier for researchers and developers to access and utilize these resources.

With 'awesome-prompt-dataset', you can accelerate your research and development in NLP and unlock new opportunities for innovation. Let's explore the possibilities together!"

<a id="whats-new-20232026"></a>
# What's New (2023–2026)

The instruction-tuning landscape has evolved rapidly. Below are representative high-quality datasets that have emerged since this list was first created, covering general SFT, RLHF/preference, DPO, code/math reasoning, agent/tool use, and Chinese resources. For Chinese datasets, also see README_zh.md for ModelScope links.

Notable additions:
- **SFT**: LIMA, UltraChat, OpenOrca/SlimOrca, OpenHermes 2.5, WizardLM/Evol-Instruct, MetaMathQA, Airoboros, Platypus, No Robots, Magpie, Infinity Instruct, Deita, WildChat, Capybara, Tulu 3 SFT Mix, Smol-Smoltalk, PersonaHub, OpenMathInstruct-1, DART-Math, LIMO, Code-Feedback, FineWeb-Edu, OpenCodeInstruct, **Nemotron-Cascade-2-SFT-Data, KIMI-K2.5-1000000x, CodeX-7M-Non-Thinking, Dolci-Instruct-SFT, AgentTrove, ToolMind, AM-Thinking-v1-Distilled, SYNTHETIC-2-SFT-verified, OpenThoughts-114k / OpenThoughts3-1.2M, OpenR1-Math-220k, NuminaMath-CoT, Bespoke-Stratos-17k**.
- **RLHF / Preference**: Anthropic HH-RLHF, Stanford SHP, UltraFeedback, HelpSteer/HelpSteer2/HelpSteer2-Preference, NVIDIA HelpSteer3-Preference, **Nemotron-Cascade-2-RL-Data, TaskTrove, SYNTHETIC-2-RL, UltraInteract_preference, RLSTACK**.
- **DPO**: Orca-DPO-Pairs, UltraFeedback-Binarized, DPO-Mix-7K, Llama3-UltraFeedback-ArmoRM, ChatML-DPO-Pairs, Magpie-Air-DPO, Magpie-Pro-DPO, Tulu 3 Preference, **RLSTACK, UltraInteract_preference, Ling-Coder-DPO, DAPO-Math-17k**.
- **Code / Math**: MetaMathQA, OpenMathInstruct-1, DART-Math, LIMO, Code-Feedback, OpenCodeInstruct, **CodeX-7M-Non-Thinking, OpenR1-Math-220k, NuminaMath-CoT, Bespoke-Stratos-17k, DAPO-Math-17k**.
- **Agent / Tool Use**: **AgentTrove, ToolMind, Nemotron-Cascade-2-SFT-Data (agentic subset)**.
- **Chinese / ModelScope**: COIG-CQIA, MSAgent, DeepCtrl-SFT, Infinity Instruct, Alpaca-GPT4 Chinese, HC3-Chinese, Magpie-Qwen2-Pro-200K-Chinese, Chinese-DeepSeek-R1-Distill-110k-SFT, Seq-Monkey, **smoltalk-chinese, OpenThoughts3-1.2M**.

The list is now organized into **SFT datasets**, **RLHF datasets**, and **DPO datasets** sections.

<a id="quick-index-by-category"></a>
# Quick Index by Category

| Category | Where to find | Highlights |
| :------- | :------------ | :----------- |
| **General SFT** | [SFT Statistics](#sft-statistics) • [General SFT Details](#general-sft) | Alpaca, LIMA, UltraChat, OpenOrca, OpenHermes 2.5, WizardLM, Magpie, No Robots, Deita, WildChat, Capybara, Tulu 3, Smol-Smoltalk, PersonaHub, Nemotron-Cascade-2, Dolci-Instruct-SFT, KIMI-K2.5-1000000x |
| **Code & Math** | [SFT Statistics](#sft-statistics) • [Code & Math Details](#code--math) | Code Alpaca, MetaMathQA, OpenMathInstruct-1, DART-Math, LIMO, Code-Feedback, OpenCodeInstruct, CodeX-7M-Non-Thinking, OpenR1-Math-220k, NuminaMath-CoT, Bespoke-Stratos-17k |
| **Multilingual & Chinese** | [SFT Statistics](#sft-statistics) • [Multilingual & Chinese Details](#multilingual--chinese) | Chinese-LLaMA-Alpaca, BELLE, COIG, COIG-CQIA, MSAgent, DeepCtrl-SFT, Infinity Instruct, Chinese-DeepSeek-R1-Distill, Seq-Monkey |
| **Agent & Tool Use** | [SFT Statistics](#sft-statistics) • [Agent & Tool Use Details](#agent--tool-use) | AgentTrove, ToolMind, MSAgent |
| **RLHF / Preference** | [RLHF Statistics](#rlhf-statistics) • [RLHF Details](#rlhf-dataset-details) | HH-RLHF, SHP, UltraFeedback, HelpSteer family, Nemotron-Cascade-2-RL, TaskTrove, SYNTHETIC-2-RL |
| **DPO** | [DPO Statistics](#dpo-statistics) • [DPO Details](#dpo-dataset-details) | Orca-DPO-Pairs, UltraFeedback-Binarized, DPO-Mix-7K, Llama3-UltraFeedback-ArmoRM, Magpie DPO, RLSTACK, UltraInteract_preference, Ling-Coder-DPO |
| **Chinese ModelScope** | README_zh.md | COIG-CQIA, MSAgent, DeepCtrl-SFT, Infinity Instruct, Chinese-DeepSeek-R1-Distill, Seq-Monkey, smoltalk-chinese, OpenThoughts3-1.2M |

<a id="how-to-use-this-list"></a>
## How to use this list
Each dataset is tagged by **Language** (`EN` = English, `CN` = Chinese, `ML` = multilingual), **Task** (`MT` = multi-task, `TS` = task-specific), and **Generation** (`HG` = human-generated, `SI` = self-instruct, `MIX` = mixed, `COL` = collection).

- Use the **Table of Contents** or **Quick Index by Category** to jump to a section.
- Use `Ctrl+F` / `Cmd+F` to search for a dataset name, model, or paper.
- Tables link directly to the dataset card; detail sections provide size, license, source model, and paper links.

<a id="sft-datasets"></a>
# SFT Datasets

Referring to [this](https://github.com/yaodongC/awesome-instruction-dataset) ([@yaodongC](https://github.com/yaodongC)), we labeled each collected dataset according to the following rules:

**(Lang) Lingual-Tags**:
- EN: Instruction datasets in English
- CN: Instruction datasets in Chinese
- ML: [Multi-lingual] Instruction datasets in multiple languages

**(Task) Task-Tags**:
- MT: [Multi-task] Datasets containing multiple tasks
- TS: [Task-specific] Datasets tailored for specific tasks

**(Gen) Generation-method**:
- HG: [Human Generated Dataset] Datasets created by humans
- SI: [Self-Instruct] Datasets generated using self-instruct methods
- MIX: [Mixed Dataset] Dataset contains both human and machine generated data
- COL: [Collection of Dataset] Dataset made from a collection of other datasets

<a id="sft-statistics"></a>
## SFT Statistics
<a id="general-sft"></a>
### General SFT
| Project                                                      |                           Datasets                           | Org                        | Nums      | Lang  | Task  | Gen  | Type                                                         | Src                                                          | Url                                                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: | -------------------------- | :-------- | :---- | :---- | :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Chain of Thought](https://github.com/google-research/FLAN)  | [cot_data](https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data) \|[few_shot_data](https://github.com/google-research/FLAN/tree/main/flan/v2/niv2_few_shot_data) | Google                     | 74771     | EN/CN | MT    | HG   | instruct with cot reasoning                                  | annotating CoT on existing data                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chain-of-Thought) |
| [GPT4all](https://github.com/nomic-ai/gpt4all)               | [nomic-ai/gpt4all-j-prompt-generations](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations) | nomic-ai                   | 806199    | EN    | MT    | COL  | code, storys and dialogs                                     | distillation from GPT-3.5-turbo                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all) |
| [GPTeacher](https://github.com/teknium1/GPTeacher)           | [GPT-4 General-Instruct ](https://github.com/teknium1/GPTeacher/tree/main/Instruct)\|[Roleplay-Instruct](https://github.com/teknium1/GPTeacher/tree/main/Roleplay) \|[Code-Instruct ](https://github.com/teknium1/GPTeacher/tree/main/Codegen)\| [Toolformer](https://github.com/teknium1/GPTeacher/tree/main/Toolformer) | teknium1                   | 29013     | EN    | MT    | SI   | general, roleplay, toolformer                                | GPT-4 & toolformer                                           | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPTeacher) |
| [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | JosephusCheung             | 534610    | ML    | MT    | SI   | various linguistic tasks                                     | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Guanaco) |
| [alpaca](https://github.com/tatsu-lab/stanford_alpaca)       | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | tatsu-lab                  | 52002     | EN    | MT    | SI   | general instruct                                             | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca) |
| [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | yahma                      | 52k       | EN    | MT    | SI   | general instruct                                             | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca) |
| [Natural Instructions](https://github.com/allenai/natural-instructions) | [Allen AI 61 task](https://instructions.apps.allenai.org/#:~:text=Download%20Natural%2DInstructions%20%2D%20v1.1)\|[1.5k task](https://instructions.apps.allenai.org/#:~:text=Natural%2DInstructions%20%2D%20v2-,.,-x) | Allen AI                   | 5040134   | ML    | MT    | COL  | diverse nlp tasks                                            | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Natural-Instructions) |
| [prosocial dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | allenai                    | 165681    | EN    | TS    | MIX  | dialogue                                                     | GPT-3 rewrites questions + humans feedback manually          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/prosocial-dialog) |
| [finance_en](https://huggingface.co/datasets/gbharti/finance-alpaca) | [gbharti/finance-alpaca](https://huggingface.co/datasets/allenai/prosocial-dialog) |                            | 68912     | EN    | TS    | COL  | financial related qa                                         | GPT3.5                                                       | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/) |
| [instruct](https://huggingface.co/datasets/swype/instruct)   | [swype/instruct](https://huggingface.co/datasets/swype/instruct) |                            | 888969    | EN    | MT    | COL  | augmented of GPT4All, Alpaca, open-source Meta datasets      | augmentation performed using the advanced NLP tools provided by AllenAI | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instruct) |
| [webGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) | [openai/webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) | openai                     | 18994     | EN    | TS    | MIX  | information retrieval (IR) QA                                | fine-tuned GPT-3, each instruction has two outputs, select better one | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/webGPT) |
| [dolly 2.0](https://github.com/databrickslabs/dolly)         | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | databricks                 | 15015     | EN    | TS    | HG   | closed QA , summarization and etc, Wikipedia as references   | human annotated                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/dolly) |
| [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry) | [mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) | mosaicml                   | 59.3K     | EN    | TS    | HG   | This dataset is a combination of [Databrick's dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset and a filtered subset of [Anthropic's HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf). | human annotated                                              |                                                              |
| [baize](https://github.com/project-baize/baize-chatbot) 白泽 | [alpaca_chat_data.json](https://github.com/project-baize/baize-chatbot/tree/main/data) \|[medical_chat_data.json](https://github.com/project-baize/baize-chatbot/blob/main/data/medical_chat_data.json) \| [quora_chat_data.json](https://github.com/project-baize/baize-chatbot/blob/main/data/quora_chat_data.json) \|[stackoverflow_chat_data.json](https://github.com/project-baize/baize-chatbot/blob/main/data/stackoverflow_chat_data.json) | project-baize              | 653699    | EN    | MT    | COL  | a collection from Alpaca, Quora, StackOverFlow and MedQuAD questions | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/baize) |
| [hh-rlhf](https://github.com/anthropics/hh-rlhf)             | [Anthropic/hh-rlhf](https://huggingface.co/datasets/anthropic/hh-rlhf) | Anthropic                  | 284517    | EN    | TS    | MIX  | dialogue                                                     | dialog between human and RLHF models                         | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/hh-rlhf) |
| [OIG(part)](https://laion.ai/blog/oig-dataset/)              |    [laion/OIG](https://huggingface.co/datasets/laion/oig)    | laion                      | 49237     | EN    | MT    | COL  | created from various tasks, such as question and answering   | using data augmentation, human annotated datasets collection | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/OIG) |
| [camel](https://github.com/lightaime/camel) \| 骆驼          | [camel-ai/code](https://huggingface.co/datasets/camel-ai/ai_society)\|[camel-ai/biology](https://huggingface.co/datasets/camel-ai/biology) \|[camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics) \|[camel-ai/chemistry](https://huggingface.co/datasets/camel-ai/chemistry) \|[camel-ai/math](https://huggingface.co/datasets/camel-ai/math) | camel-ai                   | 760620    | EN    | MT    | SI   | Role-Playing conversations in AI Society, Code, Math, Physics, Chemistry, Biolog | gpt-3.5-turbo                                                | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/camel) |
| [FLAN-Muffin](https://huggingface.co/datasets/Muennighoff/flan) | [Muennighoff/flan](https://huggingface.co/datasets/Muennighoff/flan) |                            | 1764800   | EN    | MT    | COL  | 60 nlp tasks                                                 | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/FLAN-Muffin) |
| [ShareChat](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) | [RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) | RyokoAI                    | 1663241   | EN    | MT    | MIX  | general instruct                                             | crowdsourcing to collect conversations between people and ChatGPT (ShareGPT) | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/ShareGPT) |
| [Auto CoT](https://github.com/amazon-science/auto-cot)       | [kojima-takeshi188/zero_shot_cot/dataset](https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset) \|[kojima-takeshi188/zero_shot_cot/log](https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log) | amazon-science             |           | EN    |       |      |                                                              |                                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Auto-CoT) |
| [MOSS](https://github.com/OpenLMLab/MOSS)（复旦 Moss）       | [fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)\| [moss-003-sft-data](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data/conversations/conversation_without_plugins) | fnlp                       | 1583595   | EN/CN | SI    |      |                                                              |                                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/MOSS) |
| [ultrachat](https://github.com/thunlp/UltraChat)             | [stingning/ultrachat](https://huggingface.co/datasets/stingning/ultrachat) | thnlp                      | 28247446  | EN    |       |      |                                                              |                                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/ultrachat) |
| [StackLLaMA](https://huggingface.co/datasets/lvwerra/stack-exchange-paired) | [lvwerra/stack-exchange-paired](lvwerra/stack-exchange-paired) |                            | todo      | EN    |       | HG   |                                                              |                                                              |                                                              |
| [Self-Instruct](https://github.com/yizhongw/self-instruct)   | [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct/blob/main/data/gpt3_generations/batch_221203/all_instances_82K.jsonl) |                            | 82 K      | EN    | SI    | SI   |                                                              |                                                              |                                                              |
| [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) | [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) | Openassisent               | 100 w     |       | SI    | HG   | Zhihu data for training Open Assitant                        |                                                              |                                                              |
| [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | stanfordnlp                | 385 k     | EN    | MT    | HG   |                                                              | human preferences over responses                             |                                                              |
| [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant) | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Openassisent               | 84.4k     | EN    | MT    | HG   | OpenAssistant Conversations Dataset (OASST1)                 | human-generated, human-annotated                             |                                                              |
| [akoksal/LongForm](https://github.com/akoksal/LongForm)      | [akoksal/LongForm](https://huggingface.co/datasets/akoksal/LongForm) | akoksal/LongForm           | 30k       | EN    | SI    | HG   |                                                              | 们从现有语料库（如 C4 和维基百科）中选择一组不同的人工文档，并通过 LLM 为给定的文档生成指令。 |                                                              |
| [sail-sg/symbolic-instruction-tuning](https://github.com/sail-sg/symbolic-instruction-tuning) | [sail/symbolic-instruction-tuning](https://huggingface.co/datasets/sail/symbolic-instruction-tuning) | sail-sg                    | 800K      | ML    | SI    |      |                                                              | Human Synthetic Examples                                     |                                                              |
| 医疗问答 [michael-wzhu/PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE) | [michaelwzhu/ChatMed_Consult_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) | michael-wzhu               | 110113    | CN    | SI    |      |                                                              | 互联网上的医疗问诊问题(110,113)，反映了真实世界的不同用户/患者的医疗问诊需求。目前response都是由OpenAI `GPT-3.5`引擎回答的。 |                                                              |
| [mbzuai-nlp/LaMini-LM](https://github.com/mbzuai-nlp/LaMini-LM) | [MBZUAI/LaMini-instruction](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) | MBZUAI/LaMini-instruction  | **2.58M** | EN    | MT    | SI   |                                                              | 通过离线蒸馏从大型语言模型中提取知识                         |                                                              |
| [pCLUE](https://github.com/CLUEbenchmark/pCLUE)              |       [pCLUE](https://github.com/CLUEbenchmark/pCLUE)        |                            | 120 万    |       |       |      |                                                              |                                                              |                                                              |
| [WizardLM](https://github.com/nlpxucan/WizardLM)             | [victor123/evol_instruct_70k](https://huggingface.co/datasets/victor123/evol_instruct_70k) \|[evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | WizardLM                   | 70k-196k  | EN    | MT    | SI   | general instruct evolved for complexity                      | GPT-4 / ChatGPT evolution                                  | [download](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) |
| [LIMA](https://arxiv.org/abs/2305.11206)                     | [GAIR/lima](https://huggingface.co/datasets/GAIR/lima)       | Meta AI                    | 1k        | EN    | MT    | HG   | high-quality human-written instructions                     | Stack Exchange, wikiHow, Reddit, human authors             | [download](https://huggingface.co/datasets/GAIR/lima)        |
| [UltraChat](https://github.com/thunlp/UltraChat)             | [stingning/ultrachat](https://huggingface.co/datasets/stingning/ultrachat) \|[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | THUNLP / HuggingFaceH4     | 1.5M / 200k | EN    | MT    | SI   | multi-turn dialogue                                          | GPT-3.5-turbo generated, filtered                          | [download](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| [OpenOrca](https://github.com/Open-Orca/OpenOrca)            | [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) \|[Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | Open-Orca                  | 4.2M / 550k | EN    | MT    | SI   | reasoning traces over FLAN-style prompts                    | GPT-4 / GPT-3.5 augmented FLAN Collection                  | [download](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | Nous Research / Teknium    | ~1M       | EN    | MT    | COL  | curated mixture: SlimOrca, Evol-Instruct, Platypus, ShareGPT, etc. | collection of open-source datasets                         | [download](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-3.2) | [jondurbin/airoboros-3.2](https://huggingface.co/datasets/jondurbin/airoboros-3.2) | jondurbin                  | 58.7k     | EN    | MT    | SI   | customizable self-instruct, uncensored                       | GPT-4 self-instruct                                        | [download](https://huggingface.co/datasets/jondurbin/airoboros-3.2) |
| [Platypus](https://github.com/raunak-agarwal/instruction-datasets) | [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | garage-bAInd               | 25k       | EN    | TS    | COL  | STEM focused (math/science)                                  | merging selected datasets                                  | [download](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) |
| [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | HuggingFaceH4              | 10k       | EN    | MT    | HG   | human-curated across 10 categories                           | skilled human annotators                                   | [download](https://huggingface.co/datasets/HuggingFaceH4/no_robots) |
| [Magpie](https://github.com/magpie-align/magpie)             | [Magpie-Align/Magpie-Pro-1M-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-1M-v0.1) \|[Magpie-Qwen2-Pro-200K-Chinese](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese) | Magpie-Align               | 1M / 200k | EN/CN | MT    | SI   | alignment data synthesized from aligned LLMs                 | Llama-3 / Qwen2-Instruct auto-generated                    | [download](https://huggingface.co/datasets/Magpie-Align)    |
| [Deita](https://github.com/hkust-nlp/deita)                  | [hkust-nlp/deita-10k-v0](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0) | HKUST-NLP                  | 10k       | EN    | MT    | COL  | data-efficient high-quality alignment SFT                   | complexity + quality + diversity selection on open data    | [download](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0) |
| [WildChat](https://wildchat.allen.ai/)                       | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) \|[lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | Allen AI / LMSYS           | 1M-4.8M   | ML    | MT    | HG   | real-world human-ChatGPT / multi-model conversations        | real user interactions on Chatbot Arena & ChatGPT          | [download](https://huggingface.co/datasets/allenai/WildChat-1M) |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara)   | [LDJnr/Capybara](https://huggingface.co/datasets/LDJnr/Capybara) | LDJnr / Nous Research      | 15k       | EN    | MT    | SI   | multi-turn reasoning conversations                           | Amplify-Instruct from seed datasets                        | [download](https://huggingface.co/datasets/LDJnr/Capybara)   |
| [Tulu 3 SFT Mix](https://github.com/allenai/open-instruct)   | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | Ai2                        | ~939k     | EN    | MT    | COL  | open post-training SFT mixture                               | curated open-source instruction datasets                   | [download](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Smol-Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) \|[HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Hugging Face TB            | 460k-3.4M | EN/ML | MT    | COL  | instruction data for SmolLM family                           | collection + synthetic data                                | [download](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) |
| [PersonaHub](https://github.com/tencent-ailab/persona-hub)   | [proj-persona/PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) | Tencent AI Lab             | 200k personas / 100k+ instruct | EN/CN | MT    | SI   | persona-driven synthetic instruction data                    | 1B persona scaling for diverse synthesis                   | [download](https://huggingface.co/datasets/proj-persona/PersonaHub) |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | HuggingFaceFW              | 1.3T+ tokens | ML | MT    | COL  | high-quality educational web corpus for pretrain/SFT         | FineWeb filtered for educational content                   | [download](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | NVIDIA | 15.87M | EN/ML | MT | COL | large-scale multi-domain SFT (chat, math, science, code, agent, SWE, safety) | on-policy distillation from DeepSeek-V3.2, GPT-OSS-120B, Qwen3, Nemotron-Cascade-1 | [download](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) |
| [KIMI-K2.5-1000000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x) | [ianncity/KIMI-K2.5-1000000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x) | ianncity | 693K (~5B tokens) | EN/ML | MT | SI | reasoning traces distilled from Kimi K2.5 high-reasoning mode | Kimi K2.5 synthetic | [download](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x) |
| [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | [allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | Ai2 | 2.15M | ML | MT | COL | OLMo 3 SFT mixture (math, code, multilingual, safety, logic puzzles) | curated open data + Ai2 synthetic prompts | [download](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) |
| [AM-Thinking-v1-Distilled](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled) | [a-m-team/AM-Thinking-v1-Distilled](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled) | a-m-team (Beike) | 1.89M | CN/EN | MT | SI | verified reasoning distillation (math, code, chat, science, IF) | AM-Thinking-v1 / Qwen3 / DeepSeek-R1 + verifiers | [download](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled) |
| [SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) | [PrimeIntellect/SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) | PrimeIntellect | 105K | EN | MT | SI | verified reasoning traces (math, code, formatting, puzzles) | DeepSeek-R1-0528 + distributed verification | [download](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) |
| [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | OpenThoughts | 114K | EN | MT | SI | synthetic reasoning (math, science, code, puzzles) | DeepSeek-R1 verified | [download](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | OpenThoughts | 1.2M | EN | TS | SI | scaled reasoning dataset (math / code / science) | QwQ-32B 16× annotations | [download](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) |
<a id="sft-dataset-details"></a>
<a id="code--math"></a>
### Code & Math
| Project                                                      |                           Datasets                           | Org                        | Nums      | Lang  | Task  | Gen  | Type                                                         | Src                                                          | Url                                                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: | -------------------------- | :-------- | :---- | :---- | :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Code Alpaca](https://github.com/sahil280114/codealpaca)     | [sahil280114/codealpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) |                            | 20022     | EN    | TS    | SI   | code generation, editing, optimization                       | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/CodeAlpaca) |
| [MetaMathQA](https://github.com/meta-math/MetaMath)          | [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | MetaMath                   | 395k      | EN    | TS    | SI   | mathematical reasoning                                       | bootstrapped from GSM8K & MATH via GPT-3.5                 | [download](https://huggingface.co/datasets/meta-math/MetaMathQA) |
| [OpenCodeInstruct](https://github.com/OpenCoder-llm/OpenCoder-llm) | [OpenCoder-llm/OpenCoder-llm](https://huggingface.co/datasets/OpenCoder-llm/OpenCoder-llm) | OpenCoder / NVIDIA         | 5M        | EN    | TS    | SI   | large-scale code instruction tuning                         | synthetic code data with execution verification            | [download](https://huggingface.co/datasets/OpenCoder-llm/OpenCoder-llm) |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | [nvidia/OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | NVIDIA                     | 1.8M      | EN    | TS    | SI   | mathematical problem-solving                                 | GSM8K & MATH synthetic solutions                           | [download](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) |
| [DART-Math](https://github.com/hkust-nlp/dart-math)          | [hkust-nlp/DART-Math](https://huggingface.co/datasets/hkust-nlp/DART-Math) | HKUST-NLP                  | 600k+     | EN    | TS    | SI   | difficulty-aware math instruction tuning                    | rejection tuning from MATH/GSM8K with difficulty control     | [download](https://huggingface.co/datasets/hkust-nlp/DART-Math) |
| [LIMO](https://github.com/GAIR-NLP/LIMO)                     | [GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO)       | GAIR-NLP                   | 817       | EN    | TS    | COL  | high-quality reasoning elicitation                          | curated MATH/AIME/NuminaMath with expert CoT               | [download](https://huggingface.co/datasets/GAIR/LIMO)        |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) \|[m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | M-A-P                      | 200k+     | EN    | TS    | COL  | code instruction with execution feedback                     | collection of code datasets with test feedback             | [download](https://huggingface.co/datasets/m-a-p/Code-Feedback) |
| [CodeX-7M-Non-Thinking](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking) | [Modotte/CodeX-7M-Non-Thinking](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking) | Modotte | 7.36M | EN | TS | SI | large curated code instruction pairs without reasoning chains | Modotte curation + synthetic generation | [download](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking) |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Hugging Face | 220K | EN | TS | SI | math reasoning traces over NuminaMath 1.5 | DeepSeek R1 + Math-Verify / Llama judge | [download](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | AI-MO | 860K | EN/CN | TS | COL | math problems with CoT solutions (K-12 to olympiad) | OCR + translation + realignment | [download](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | Bespoke Labs | 17K | EN | TS | SI | high-quality reasoning distillation (math / code / science) | DeepSeek-R1 rejection sampling | [download](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |

<a id="multilingual--chinese"></a>
### Multilingual & Chinese
| Project                                                      |                           Datasets                           | Org                        | Nums      | Lang  | Task  | Gen  | Type                                                         | Src                                                          | Url                                                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: | -------------------------- | :-------- | :---- | :---- | :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)    | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Hello-SimpleAI \| 万得资讯 | 37175     | EN/CN | TS    | MIX  | dialogue evaluation                                          | human or ChatGPT                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/HC3) |
| [HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | Hello-SimpleAI\|万得资讯   | 13k       | CN    | TS    | MIX  | dialogue evaluation                                          | human or ChatGPT                                             |                                                              |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [alpaca_data_zh_51k](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/data/alpaca_data_zh_51k.json) | ymcui(讯飞)                | 51k       | CN    | MT    | SI   | general instruct                                             | text-davinci-003                                             |                                                              |
| [Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM)  骆驼 | [trans_chinese_alpaca_data](https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/data/trans_chinese_alpaca_data.json) | LC1332(商汤)               | 52k       | CN    | MT    | SI   | general instruct                                             | text-davinci-003                                             |                                                              |
| [belle_cn](https://huggingface.co/BelleGroup)                | [BelleGroup/train_1M_CN](https://huggingface.co/datasets/bellegroup/train_1M_CN) \|[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/bellegroup/train_0.5M_CN) | BelleGroup(链家)           | 1079517   | CN    | TS/MT | SI   | general, mathematical reasoning, dialogue                    | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/belle_cn) |
| [instinwild](https://github.com/XueFuzhao/InstructionWild)   | [instinwild_ch](https://github.com/XueFuzhao/InstructionWild/tree/main/data) \| [instinwild_en](https://github.com/XueFuzhao/InstructionWild/tree/main/data) |                            | 52191     | EN/CN | MT    | SI   | generation, open-qa, mind-storm                              | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instinwild) |
| [华驼(HuaTuo)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | [中文医学知识](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/data/llama_data.json) \|[肝癌](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/data-literature/liver_cancer.json) | SCIR-HI(哈工大)            | 8K        | CN    | TS    | SI   | 公开和自建的中文医学知识库                                   | GPT3.5                                                       |                                                              |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)        | [bigscience/xP3](https://huggingface.co/datasets/bigscience/xP3) | bigscience                 | 78883588  | ML    | MT    | COL  | a collection of prompts & datasets across 46 of languages & 16 NLP tasks | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/xP3) |
| [firefly](https://github.com/yangjianxin1/Firefly)           | [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) |                            | 1649398   | CN    | MT    | COL  | 23 nlp tasks                                                 | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/firefly) |
| [Alpaca_GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | [alpaca_gpt4_data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)\|[alpaca_gpt4_data_zh](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json) \|[comparison_data_v2](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/comparison_data_v2.json) | 微软                       | 52002     | EN/CN | MT    | SI   | general instruct                                             | generated by GPT-4 using Alpaca                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpacaGPT4) |
| [GAOKAO](https://github.com/OpenLMLab/GAOKAO-Bench)          | [Fill-in-the-blank_Questions](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data/Fill-in-the-blank_Questions) \| [Multiple-choice_Questions](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data/Multiple-choice_Questions) \| [Open-ended_Questions](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data/Open-ended_Questions) | OpenLMLab                  | 2785      | CN    | MT    | COL  | Multiple-choice, Fill-in-the-blank and Open-ended questions from examination | human annotated                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GAOKAO) |
| [COIG](https://huggingface.co/datasets/BAAI/COIG)            |      [COIG](https://huggingface.co/datasets/BAAI/COIG)       | BAAI\|智源                 | 298428    | CN    | MT    | COL  | collect fron Exam, Translated, Human Value Alignment Instructions and Counterfactural Correction Multi-round Chat | using automatic tool and manual verification                 | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/COIG) |
| [Infinity Instruct](https://github.com/FlagOpen/Infinity-Instruct) | [AI-ModelScope/Infinity-Instruct](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct) | BAAI / FlagOpen            | 7M+       | EN/CN | MT    | COL  | large-scale general instruction collection                   | collection + synthesis                                     | [download](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct) |
| [COIG-CQIA](https://github.com/CSJianYang/COIG-CQIA)         | [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) \|[modelscope/COIG-CQIA](https://modelscope.cn/datasets/m-a-p/COIG-CQIA) | 01.AI / M-A-P              | 48k       | CN    | MT    | COL  | high-quality Chinese Q&A and articles                        | Chinese internet data, cleaned + manually reviewed         | [download](https://huggingface.co/datasets/m-a-p/COIG-CQIA) |
| [DeepCtrl-SFT](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data) | [deepctrl/deepctrl-sft-data](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data) | DeepCtrl                   | 12M       | CN/EN | MT    | MIX  | safe general-domain SFT data                                 | curated + safety-filtered Chinese/English data             | [download](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data) |
| [Chinese-DeepSeek-R1-Distill-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k) | [Congliu/Chinese-DeepSeek-R1-Distill-data-110k](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k) \|[liucong/Chinese-DeepSeek-R1-Distill-data-110k](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k) | Cong Liu                   | 110k      | CN    | MT    | SI   | Chinese reasoning SFT distilled from DeepSeek-R1-671B       | math/exam/STEM/general Chinese reasoning                   | [download](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k) |
| [Seq-Monkey](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey) | [ddzhu123/seq-monkey](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey) | Mobvoi / 出门问问          | 10B tokens | CN   | MT    | COL  | Chinese pre-training corpus (general/web/QA/code)           | Chinese web, encyclopedia, Q&A, blogs, books, code       | [download](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey) |

<a id="agent--tool-use"></a>
### Agent & Tool Use
| Project                                                      |                           Datasets                           | Org                        | Nums      | Lang  | Task  | Gen  | Type                                                         | Src                                                          | Url                                                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: | -------------------------- | :-------- | :---- | :---- | :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)        | [gpt4tools_71k.json](https://drive.google.com/file/d/1JKIT-Or1of7TJuWvmrJpPoOx0cLdcWry/view?usp=share_link) | StevenGrove                | 71446     | EN    | MT    | SI   | a collection of tool-related instructions                    | gpt-3.5-turbo                                                | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/gpt4tools) |
| [MSAgent / MSAgent-Bench](https://github.com/modelscope/modelscope-agent) | [iic/MSAgent-Bench](https://huggingface.co/datasets/iic/MSAgent-Bench) \|[modelscope/MSAgent-Bench](https://modelscope.cn/datasets/iic/MSAgent-Bench) | Alibaba / ModelScope       | 600k+     | CN/EN | MT    | COL  | tool-use and agent capabilities                              | API calling, multi-turn tool interactions                  | [download](https://huggingface.co/datasets/iic/MSAgent-Bench) |
| [AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | OpenThoughts | 1.7M | EN | MT | COL | agentic interaction traces (code repair, shell, math, CP, computer-use) | 219 source datasets via terminus-2 / Harbor | [download](https://huggingface.co/datasets/open-thoughts/AgentTrove) |
| [ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) | [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) | Nanbeige Lab | 369K | EN | TS | MIX | reasoning-enhanced function-calling / tool-use trajectories | multi-agent simulation + curated open data | [download](https://huggingface.co/datasets/Nanbeige/ToolMind) |


#### SFT Dataset Details

<a id="general-sft"></a>
### General SFT
#### [Alpaca -Stanford](https://github.com/tatsu-lab/stanford_alpaca)

- [Paper/Project Link](https://github.com/tatsu-lab/stanford_alpaca)
- [Dataset Link](https://github.com/tatsu-lab/stanford_alpaca)
- Data generation model: text-davinci-003
- Cost: $600

The Alpaca of the Stanford release is a fine-tuning model for instruct-tuning based on the Meta Ai LLaMA model.

Alpaca automatically generated 52k instruction data using GPT-3.5 and used it to fine-tune the LLaMA model. Experimental results show that it can reach or even exceed the performance of GPT-3.5 on some tasks.

#### [Instruction in the Wild](https://github.com/XueFuzhao/InstructionWild)

- [Paper/Project Link](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat)
- [Dataset Link](https://github.com/XueFuzhao/InstructionWild)
- Data generation model: text-davinci-003

Instruction Tuning is a key component of ChatGPT. OpenAI used their user-based Instruction dataset, but unfortunately, this dataset is not open-sourced. Self-Instruct released a small instruction dataset including 175 instructions written by human labors. Standford Alpaca Team generated 52K instructions by text-davinci-003 model based on the the 175 seed instructions above.

This project targets on a larger and more diverse instruction dataset. To this end, we collected 429 instructions from ChatGPT usage screenshots and released both English and Chinese versions. We found these instructions are very diverse even if the scale is still small. We follow Alpaca to generate 52K instructions and their responses. All data can be found in data dir.

Note: This is an ongoing project. We are still collecting and improving our data. We release this dataset as early as possible to speedup our LLM research. We will also release a whitepaper soon.

#### [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

- Data generation model: text-davinci-003
- Cost: $6000

52K instruction data generated from modified self-instruct pipeline with human written 429 seed task.


#### [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)

- [DataLinks](https://huggingface.co/datasets/stanfordnlp/SHP)

SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)).

Each example is a Reddit post with a question/instruction and a pair of top-level comments for that post, where one comment is more preferred by Reddit users (collectively). SHP exploits the fact that if comment A was written after comment B but has a higher score nonetheless, then A is ostensibly more preferred to B. If A had been written before B, then we could not conclude this, since its higher score could have been the result of more visibility. We chose data where the preference label is intended to reflect which response is more helpful rather than which is less harmful, the latter being the focus of much past work.

How is SHP different from [Anthropic's HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)? Most notably, all the data in SHP is naturally occurring and human-written, whereas the responses in HH-RLHF are machine-written, giving us two very different distributions that can complement each other.


#### [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

- Summary:The the first human-ChatGPT comparison corpus (English Version), named HC3 dataset
- Data generation model: `gpt-3.5`, `human generated`
- paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
- Cost: N/A

#### [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

- Summary:The the first human-ChatGPT comparison corpus (Chinese Version), named HC3 dataset
- Data generation model: `gpt-3.5`, `human generated`
- paper: [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)
- Cost: N/A


#### [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)

- Summary: ProsocialDialog is the first large-scale multi-turn English dialogue dataset to teach conversational agents to respond to problematic content following social norms.
- Data generation model: `gpt-3.5`, `human generated`
- paper: [ProsocialDialog: A Prosocial Backbone for Conversational Agents](https://arxiv.org/abs/2205.12688)
- Cost: N/A

#### [allenai/natural-instructions](https://github.com/allenai/natural-instructions)

- Summary: A community effort to create a large collection of `1,616 diverse NLP tasks` and their natural language definitions/instructions.
- Data generation model: `Human generated`
- paper: [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705)
- Cost: N/A


#### [PhoebusSi/Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)

- Summary: A datset for Chain-of-Thoughts reasoning based on LLaMA and Alpaca. Note: Their repository will continuously collect various instruction tuning datasets. [Github Repo](https://github.com/PhoebusSi/Alpaca-CoT)
- paper: N/A
- Cost: N/A

#### [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)

- Summary: gpt4all leverages three publicly available datasets: 1.[laion/OIG](https://huggingface.co/datasets/laion/OIG), 2.[pacovaldez/stackoverflow-questions](https://huggingface.co/datasets/pacovaldez/stackoverflow-questions) 3. subset of [bigscience/bloomz-p3](https://huggingface.co/bigscience/bloomz-p3)
- Data generation model: N/A
- paper: [GPT4All: Training an Assistant-style Chatbot with Large Scale Data Distillation from GPT-3.5-Turbo](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf)
- Cost: $500


#### [bigscience/xP3](https://huggingface.co/datasets/bigscience/xP3)

- Summary: [Prompt-resource] xP3 (Crosslingual Public Pool of Prompts) is a collection of prompts & datasets across 46 of languages & 16 NLP tasks.
- Data generation model: N/A
- paper: [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
- Cost: N/A



#### [teknium1/GPTeacher](https://github.com/teknium1/GPTeacher)

- Summary: A collection of modular datasets generated by GPT-4, General-Instruct - Roleplay-Instruct - Code-Instruct - and Toolformer
- Data generation model: `GPT-4`
- paper: N/A
- Cost: N/A

#### [thunlp/UltraChat](https://github.com/thunlp/UltraChat)

- Summary: UltraChat aims to construct an open-source, large-scale, and multi-round dialogue data. The first part of UltraChat (i.e., the Questions about the World sector) is released, which contains 280k diverse and informative dialogues. More dialogues about writing and creation, assistance on existing materials are to come.
- Data generation model: `GPT-3.5-turbo`
- paper: N/A
- Cost: N/A

#### [cascip/ChatAlpaca](https://github.com/cascip/ChatAlpaca)

- Summary: Based on the Stanford Alpaca data, ChatAlpaca extends the data to multi-turn instructions and their corresponding responses. More data (20k) and the Chinese translated version are to come.
- Data generation model: `GPT-3.5-turbo`
- paper: N/A
- Cost: N/A
- Related: [(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)

#### [YeungNLP/firefly-train-1.1M)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- Summary: Chinese datasets of 23 tasks combined with human-written instruction templates.
- Data generation model: N/A
- paper: N/A
- Cost: N/A

#### [orhonovich/unnatural-instructions](https://github.com/orhonovich/unnatural-instructions)
- Summary: 64K examples by prompting a language model with three seed examples of instructions and eliciting a fourth. Then the set is expanded to 240K by prompting the model to rephrase each instruction.
- Data generation model: `text-davinci-002`
- paper: [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)
- Cost: N/A

#### [Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- Summary: 52K instruction-following data generated by GPT-4 with the original Alpaca prompts & Alpaca prompts translated into Chinese by ChatGPT + 9K instruction-following data generated by GPT-4 with prompts in Unnatural Instruction.
- Data generation model: `GPT-4`
- paper: [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)
- Cost: N/A
- Related:
    -[(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)
    -[(orhonovich/unnatural-instructions)|240K|EN|MT|MIX](https://github.com/orhonovich/unnatural-instructions)

#### [databrickslabs/dolly](https://github.com/databrickslabs/dolly/tree/master/data)
- Summary: This datset was generated by thousands of Databricks employees in several of the behavioral categories outlined in the InstructGPT paper, including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization.
- Data generation model: N/A
- paper: [Free Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- Cost: N/A

#### [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- Summary: OpenAssistant Conversations (OASST1), a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages distributed across 66,497 conversation trees, in 35 different languages, annotated with 461,292 quality ratings.
- Data generation model: N/A
- paper: [OpenAssistant Conversations - Democratizing Large Language Model Alignment](https://drive.google.com/file/d/10iR5hKwFqAKhL3umx8muOWSRm7hs5FqX/view)
- Cost: N/A

#### BELLE/data/1.5M

- 下载地址: [https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)
- 数据量: 1.5M
- 生成方式: self-instruct，使用了中文种子任务，以及openai的text-davinci-003接口
- 涉及任务: 包含175个种子任务，[https://github.com/LianjiaTech/BELLE/blob/main/data/1.5M/zh_seed_tasks.json](https://github.com/LianjiaTech/BELLE/blob/main/data/1.5M/zh_seed_tasks.json)
- 数据示例: [https://huggingface.co/datasets](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

#### alpaca_chinese_dataset

- 下载地址: [https://github.com/hikariming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset)
- 数据量: 52k
- 生成方式: 借助chatgpt对原始的[stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)做机器翻译，并加入人工校验来保证质量
- 涉及任务: 与原始的[stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)一致，可以在原项目的[seed_task.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl)中查到全部任务

#### Med-ChatGLM/data

- 下载地址: [https://github.com/SCIR-HI/Med-ChatGLM](https://github.com/SCIR-HI/Med-ChatGLM)
- 数据量: 7k
- 生成方式: 利用GPT3.5接口围绕医学知识库构建问答数据，并设置了多种Prompt形式来充分利用知识
- 涉及任务: 医学领域相关的问答，包含并发症，高危因素，组织学检查，临床症状，药物治疗，辅助治疗

#### pCLUE

- 下载地址: [https://github.com/CLUEbenchmark/pCLUE](https://github.com/CLUEbenchmark/pCLUE)
- 数据量: 1.2M
- 生成方式: 通过原有的NLP任务数据集，结合特定的[prompt](https://www.zhihu.com/search?q=prompt&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"624084039"})模板生成
- 涉及任务: 包含9个NLP数据集，涉及的NLP任务有[文本分类](https://www.zhihu.com/search?q=文本分类&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"624084039"})/自然语言推理/语义匹配/[指代消解](https://www.zhihu.com/search?q=指代消解&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"624084039"})/关键词识别/阅读理解

#### COIG

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

#### [LIMA](https://arxiv.org/abs/2305.11206)
- [Dataset Link](https://huggingface.co/datasets/GAIR/lima)
- Summary: 1,000 carefully curated human-written instructions showing that "less is more for alignment". LIMA demonstrates that short fine-tuning on small, high-quality data can achieve competitive results with much larger synthetic datasets.
- Data generation model: `human generated`
- paper: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- Cost: N/A

#### [UltraChat / UltraChat 200k](https://github.com/thunlp/UltraChat)
- [Dataset Link](https://huggingface.co/datasets/stingning/ultrachat)
- [Filtered Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- Summary: Large-scale multi-turn dialogue dataset generated by GPT-3.5-turbo. The HuggingFaceH4 release provides a sanitized, filtered 200k subset used in models like Zephyr.
- Data generation model: `GPT-3.5-turbo`
- paper: [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233)
- Cost: N/A

#### [OpenOrca / SlimOrca](https://github.com/Open-Orca/OpenOrca)
- [Dataset Link](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [Cleaned Subset Link](https://huggingface.co/datasets/Open-Orca/SlimOrca)
- Summary: ~4.2M examples augmenting FLAN Collection prompts with GPT-4/GPT-3.5 completions and reasoning traces. SlimOrca is a cleaned, deduplicated subset focused on replicating Microsoft Orca-style reasoning.
- Data generation model: `GPT-4`, `GPT-3.5`
- paper: [OpenOrca: An Open Dataset of GPT Augmented FLAN Reasoning Traces](https://arxiv.org/abs/2308.07116)
- Cost: N/A

#### [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [Dataset Link](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- Summary: A ~1M sample curated mixture of high-quality open-source instruction datasets including SlimOrca, Evol-Instruct, Platypus, ShareGPT, Airoboros, GPTeacher and others. Widely used for training generalist chat models.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

#### [WizardLM / Evol-Instruct](https://github.com/nlpxucan/WizardLM)
- [Dataset Link 70k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k)
- [Dataset Link V2 196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k)
- Summary: Evol-Instruct iteratively rewrites simple instructions into more complex variants using LLMs. WizardLM applies this method to create complexity-evolved instruction data for training capable instruction followers.
- Data generation model: `GPT-4`, `ChatGPT`
- paper: [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
- Cost: N/A

#### [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-3.2)
- [Dataset Link](https://huggingface.co/datasets/jondurbin/airoboros-3.2)
- Summary: Customizable self-instruct pipeline producing high-quality instruction data across general, code, roleplay and other domains. Version 3.2 contains ~58.7k samples.
- Data generation model: `GPT-4`
- paper: N/A
- Cost: N/A

#### [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- Summary: 10k high-quality, fully human-curated instruction dataset modeled after OpenAI's InstructGPT paper. Covers 10 categories including generation, QA, coding, summarization and more.
- Data generation model: `human generated`
- paper: N/A
- Cost: N/A

#### [Magpie](https://github.com/magpie-align/magpie)
- [Dataset Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-1M-v0.1)
- [Chinese Dataset Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese)
- Summary: Alignment data synthesis from scratch by prompting aligned LLMs with nothing—no seed data required. Magpie-Pro contains ~1M samples generated from Llama-3-70B-Instruct; a 200k Chinese variant is available from Qwen2.
- Data generation model: `Llama-3-Instruct`, `Qwen2-Instruct`
- paper: [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
- Cost: N/A

#### [Deita](https://github.com/hkust-nlp/deita)
- [Dataset Link](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0)
- Summary: A 10K-sample data-efficient instruction tuning dataset selected by complexity, quality, and diversity scorers. Deita shows that principled data selection can match or exceed models trained on 10x more data.
- Data generation model: `COL`
- paper: [What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685)
- Cost: N/A

#### [WildChat](https://wildchat.allen.ai/)
- [Dataset Link](https://huggingface.co/datasets/allenai/WildChat-1M)
- [Related Dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- Summary: Large-scale real-world human-ChatGPT / multi-model conversation logs (1M–4.8M). WildChat captures diverse, often longer and more complex user prompts than synthetic datasets; LMSYS-Chat-1M covers 25+ models and ~154 languages.
- Data generation model: `HG`
- paper: [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.14770) / [LMSYS-Chat-1M](https://arxiv.org/abs/2309.11998)
- Cost: N/A

#### [Capybara](https://huggingface.co/datasets/LDJnr/Capybara)
- [Dataset Link](https://huggingface.co/datasets/LDJnr/Capybara)
- Summary: ~15K multi-turn reasoning conversations generated using Amplify-Instruct, expanding high-quality seed instructions into deep, multi-turn dialogues across science, math, logic, and culture.
- Data generation model: `GPT-4`
- paper: N/A
- Cost: N/A

#### [Tulu 3 SFT Mix](https://github.com/allenai/open-instruct)
- [Dataset Link](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- Summary: ~939K open post-training SFT mixture curated by Ai2 for Tulu 3. Combines high-quality public instruction datasets into a single training blend.
- Data generation model: `COL`
- paper: [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.09601)
- Cost: N/A

#### [Smol-Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Dataset Link 2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- Summary: Instruction dataset for the SmolLM family. smoltalk2 (~3.4M samples) adds OpenThoughts, Tulu 3, and multilingual data for training small but capable chat models.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

#### [PersonaHub](https://github.com/tencent-ailab/persona-hub)
- [Dataset Link](https://huggingface.co/datasets/proj-persona/PersonaHub)
- Summary: 200K personas (toward 1B) released by Tencent AI Lab for persona-driven synthetic data generation. Configurations include instruction, math, reasoning, knowledge, NPC, and tool definitions.
- Data generation model: `SI`
- paper: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)
- Cost: N/A

#### [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
- [Dataset Link](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
- [Paper](https://arxiv.org/abs/2603.19220)
- Summary: ~15.87M-sample multi-domain SFT dataset used to train Nemotron-Cascade-2-30B-A3B. Covers chat (~9.3M), math (~2.9M), science (~1.8M), instruction following (~820K), conversational agent (~548K), terminal agent (~324K), SWE (~102K), and safety. Responses were generated via on-policy distillation from DeepSeek-V3.2, GPT-OSS-120B, Qwen3-235B-A22B, and Nemotron-Cascade-1.
- Data generation model: `DeepSeek-V3.2`, `GPT-OSS-120B`, `Qwen3-235B-A22B`, `Nemotron-Cascade-1`
- License: NVIDIA Open Model License

#### [KIMI-K2.5-1000000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x)
- [Dataset Link](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x)
- Summary: ~693K reasoning traces (~5B tokens) distilled from Moonshot AI's Kimi K2.5 in high-reasoning mode. Composition is roughly 50% code, 20% science, 15% math, plus CS, logic, creative writing, and multilingual STEM subsets.
- Data generation model: `Kimi K2.5`
- License: Apache 2.0

#### [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)
- [Dataset Link](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)
- [Paper](https://arxiv.org/abs/2512.13961)
- Summary: 2.15M instruction samples used to train OLMo 3 7B Instruct SFT. Mixes existing sources (OpenThoughts 3, FLAN v2, Aya, Tulu 3 personas, WildChat upgraded with GPT-4.1, SciRIFF, etc.) with new Ai2 prompts for precise instruction following, Python algorithms, logic puzzles, verifiable reasoning, and tool use.
- Data generation model: `COL`
- License: ODC-BY

#### [AM-Thinking-v1-Distilled](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled)
- [Dataset Link](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled)
- [Paper](https://arxiv.org/abs/2505.14464)
- Summary: 1.89M verified reasoning examples distilled from AM-Thinking-v1 (and also Qwen3-235B-A22B / DeepSeek-R1). Covers math (~29.5%), code (~17.1%), general chat (~41.8%), and science/IF/dialogue. Each response is verified with Math-Verify, sandbox execution, and model judges.
- Data generation model: `AM-Thinking-v1`, `Qwen3-235B-A22B`, `DeepSeek-R1`
- License: research-only

#### [SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified)
- [Dataset Link](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified)
- [Blog](https://www.primeintellect.ai/blog/synthetic-2-release)
- Summary: 105K verified reasoning traces from the SYNTHETIC-2 collection. Responses are generated by DeepSeek-R1-0528 and filtered to keep only correct solutions (reward = 1 or >0.7). Tasks include code output prediction, Pydantic/JSON formatting, sentence unscrambling, ASCII trees, and other verifiable reasoning problems.
- Data generation model: `DeepSeek-R1-0528`

#### [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- [Dataset Link](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- [Paper](https://arxiv.org/abs/2506.04178)
- Summary: 114K high-quality synthetic reasoning examples across math, science, code, and puzzles. Used to train OpenThinker-7B / 32B. Solutions are generated by DeepSeek-R1 and verified.
- Data generation model: `DeepSeek-R1`
- License: Apache 2.0

#### [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [Dataset Link](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [Paper](https://arxiv.org/abs/2506.04178)
- Summary: Third iteration of the OpenThoughts series with 1.2M examples (850K math, 250K code, 100K science). Each question is annotated 16× with QwQ-32B and rigorously filtered. Used to train OpenThinker3-7B.
- Data generation model: `QwQ-32B`
- License: Apache 2.0

<a id="code--math"></a>
### Code & Math
#### [MetaMathQA](https://github.com/meta-math/MetaMath)
- [Dataset Link](https://huggingface.co/datasets/meta-math/MetaMathQA)
- Summary: ~395K math question-answer pairs bootstrapped from GSM8K and MATH via forward/backward reasoning and rephrasing. Used to significantly improve mathematical reasoning in LLMs.
- Data generation model: `GPT-3.5`
- paper: [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284)
- Cost: N/A

#### [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)
- [Dataset Link](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)
- Summary: 1.8M math instruction tuning pairs with synthetic solutions for GSM8K and MATH, designed to improve open math reasoning.
- Data generation model: `GPT-4`
- paper: [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176)
- Cost: N/A

#### [DART-Math](https://github.com/hkust-nlp/dart-math)
- [Dataset Link](https://huggingface.co/datasets/hkust-nlp/DART-Math)
- Summary: Difficulty-Aware Rejection Tuning dataset for math. Generates and filters reasoning trajectories by difficulty, yielding 600K+ high-quality math instruction examples.
- Data generation model: `GPT-4`
- paper: [DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving](https://arxiv.org/abs/2407.13690)
- Cost: N/A

#### [LIMO](https://github.com/GAIR-NLP/LIMO)
- [Dataset Link](https://huggingface.co/datasets/GAIR/LIMO)
- Summary: Only 817 curated high-quality reasoning examples that elicit strong complex reasoning (AIME 57%, MATH 95%) from a 32B base model, demonstrating "less is more for reasoning".
- Data generation model: `COL`
- paper: [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)
- Cost: N/A

#### [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)
- [Dataset Link](https://huggingface.co/datasets/m-a-p/Code-Feedback)
- [Filtered Version](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)
- Summary: 200K+ code instruction pairs with execution feedback from the M-A-P project. Used to train code models with test-driven verification signals.
- Data generation model: `COL`
- paper: [M-A-P: Introducing High-Quality Code-Feedback Dataset](https://arxiv.org/abs/2407.12883)
- Cost: N/A

#### [OpenCodeInstruct](https://github.com/OpenCoder-llm/OpenCoder-llm)
- [Dataset Link](https://huggingface.co/datasets/OpenCoder-llm/OpenCoder-llm)
- Summary: 5 million code instruction tuning samples with execution-based verification, one of the largest open code instruction datasets.
- Data generation model: `COL`
- paper: [OpenCoder: The Open Cookbook for Top-Tier Code LLMs](https://arxiv.org/abs/2411.04905)
- Cost: N/A

#### [CodeX-7M-Non-Thinking](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking)
- [Dataset Link](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking)
- Summary: 7.36M curated code instruction pairs emphasizing direct code solutions (no step-by-step reasoning chains). Covers Python, Java, C++, JavaScript, algorithms, web dev, ML/AI, databases, and competitive programming.
- Data generation model: `COL` / Modotte curation
- License: Apache 2.0

#### [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [Dataset Link](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [GitHub](https://github.com/huggingface/open-r1)
- Summary: 220K math problems from NuminaMath 1.5 with 2–4 reasoning traces per problem generated by DeepSeek R1. Traces are verified with Math-Verify (~88%) and Llama-3.3-70B-Instruct as judge (~12%). Includes `default` (~94K) and `extended` (~131K) splits.
- Data generation model: `DeepSeek-R1`
- License: Apache 2.0

#### [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [Dataset Link](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- Summary: ~860K math problems with Chain-of-Thought solutions, spanning Chinese K-12 exercises to US/international olympiad problems. Sources include cn_k12, olympiads, AoPS, GSM8K, MATH, and synthetic data.
- Data generation model: `COL`
- License: Apache 2.0

#### [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- [Dataset Link](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- [Blog](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation)
- Summary: 17K high-quality reasoning-distillation examples (5K code from APPs/TACO, 10K math from AIME/MATH/Olympiads, 1K science/puzzles from STILL-2). Uses DeepSeek-R1 as teacher and gpt-4o-mini for filtering, improving correct-solution retention to ~73%.
- Data generation model: `DeepSeek-R1`
- License: Apache 2.0

<a id="multilingual--chinese"></a>
### Multilingual & Chinese
#### [xP3](https://huggingface.co/datasets/bigscience/xP3)
- [Dataset Link](https://huggingface.co/datasets/bigscience/xP3)
- Summary: [Prompt-resource] xP3 (Crosslingual Public Pool of Prompts) is a collection of prompts & datasets across 46 of languages & 16 NLP tasks.
- Data generation model: N/A
- paper: [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
- Cost: N/A

#### [firefly](https://github.com/yangjianxin1/Firefly)
- [Dataset Link](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- Summary: Chinese datasets of 23 tasks combined with human-written instruction templates.
- Data generation model: N/A
- paper: N/A
- Cost: N/A

#### [COIG](https://huggingface.co/datasets/BAAI/COIG)
- [Dataset Link](https://huggingface.co/datasets/BAAI/COIG)
- Summary: Chinese Open Instruction Generalist dataset containing translated instructions, exam instructions, value alignment instructions, counterfactual correction multi-round chat, and Leetcode instructions.
- Data generation model: `COL`
- paper: [Chinese Open Instruction Generalist: A Preliminary Release](https://arxiv.org/abs/2304.07987)
- Cost: N/A

#### [Infinity Instruct](https://github.com/FlagOpen/Infinity-Instruct)
- [Dataset Link (ModelScope)](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct)
- [Dataset Link (HuggingFace)](https://huggingface.co/datasets/AI-ModelScope/Infinity-Instruct)
- Summary: Large-scale general instruction dataset (7M+) released by BAAI / FlagOpen for both Chinese and English instruction tuning.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

#### [COIG-CQIA](https://github.com/CSJianYang/COIG-CQIA)
- [Dataset Link](https://huggingface.co/datasets/m-a-p/COIG-CQIA)
- [ModelScope Link](https://modelscope.cn/datasets/m-a-p/COIG-CQIA)
- Summary: High-quality Chinese instruction fine-tuning dataset following the "quality is all you need" philosophy. Built from Chinese internet Q&A and articles with deep cleansing, restructuring, and manual review.
- Data generation model: `COL`
- paper: [COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning](https://arxiv.org/abs/2403.18058)
- Cost: N/A

#### [MSAgent / MSAgent-Bench](https://github.com/modelscope/modelscope-agent)
- [Dataset Link](https://huggingface.co/datasets/iic/MSAgent-Bench)
- [ModelScope Link](https://modelscope.cn/datasets/iic/MSAgent-Bench)
- Summary: Chinese/English agent tool-use dataset for training LLMs to call APIs and use external tools in multi-turn conversations.
- Data generation model: `MIX`
- paper: N/A
- Cost: N/A

#### [DeepCtrl-SFT](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [ModelScope Link](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- Summary: 12M safe Chinese/English general-domain SFT entries (~3B tokens). Designed for small-scale Chinese LLM instruction fine-tuning with safety filtering.
- Data generation model: `MIX`
- paper: N/A
- Cost: N/A

#### [Chinese-DeepSeek-R1-Distill-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- [Dataset Link](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- [ModelScope Link](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k)
- Summary: 110K Chinese reasoning SFT samples distilled from the full DeepSeek-R1-671B model. Covers math (~36.5K), exam (~2.4K), STEM (~12K), and general Chinese reasoning (~58K).
- Data generation model: `DeepSeek-R1-671B`
- paper: N/A
- Cost: N/A

#### [Seq-Monkey](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey)
- [ModelScope Link](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey)
- Summary: ~10B token Chinese pre-training corpus released by Mobvoi, including web, encyclopedia, Q&A, blogs, books, and code. Mainly for pre-training/continued pre-training; often combined with instruction data for full Chinese LLM training.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

<a id="agent--tool-use"></a>
### Agent & Tool Use
#### [AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove)
- [Dataset Link](https://huggingface.co/datasets/open-thoughts/AgentTrove)
- [GitHub](https://github.com/open-thoughts/OpenThoughts-Agent)
- Summary: 1.7M agentic interaction traces drawn from 219 source datasets. Covers code repair, shell scripting, mathematical problem solving, competitive programming, and general computer-use tasks. Formatted as ShareGPT-style conversations with `original_source`, `original_teacher`, and `reward` metadata. Companion RL/evaluation dataset: [TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove).
- Data generation model: `COL` (terminus-2 / Harbor)
- License: Apache 2.0

#### [ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind)
- [Dataset Link](https://huggingface.co/datasets/Nanbeige/ToolMind)
- [Paper](https://arxiv.org/abs/2511.15718)
- Summary: ~369K reasoning-enhanced tool-use trajectories. Combines 160K synthetic multi-agent trajectories over 20K+ functions with 200K augmented open-source function-calling data. Includes two-stage quality filtering and explicit `<think>` reasoning traces.
- Data generation model: `MIX`
- License: Apache 2.0

<a id="rlhf--preference-datasets"></a>
# RLHF / Preference Datasets

<a id="rlhf-statistics"></a>
#### RLHF Statistics

|                           Project                            | Links                                                        |              Org              | Nums   |  Lang   | Summary                                                      |
| :----------------------------------------------------------: | ------------------------------------------------------------ | :---------------------------: | ------ | :-----: | ------------------------------------------------------------ |
| [webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) |                                                              |            Openai             | 19,578 | English | In the [WebGPT paper](https://arxiv.org/abs/2112.09332), the authors trained a reward model from human feedback. They used the reward model to train a long form question answering model to align with human preferences. This is the dataset of all comparisons that were marked as suitable for reward modeling by the end of the WebGPT project. There are 19,578 comparisons in total. |
|    [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)    |                                                              |          stanfordnlp          | 349 K  | English | SHP is a dataset of 385K collective human preferences over responses to questions/instructions in 18 different subject areas, from cooking to legal advice. The preferences are meant to reflect the helpfulness of one response over another, and are intended to be used for training RLHF reward models and NLG evaluation models (e.g., [SteamSHP](https://huggingface.co/stanfordnlp/SteamSHP-flan-t5-xl)). |
| [rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets) |                                                              |           yitingxie           | 76.3 k | English |                                                              |
| [Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) |                                                              |            Dahoas             | 112 k  | English | Anthropic's HH dataset reformatted into prompt, chosen, rejected samples. |
| [Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) |                                                              |            Dahoas             |        | English |                                                              |
| [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) |                                                              |            Dahoas             | 76.3k  | English | Split of [hh-static](https://huggingface.co/datasets/Dahoas/static-hh) used for training reward models after supervised fine-tuning. |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) |                                                              |           Anthropic           | 22k    | English | This RLHF dataset is an iterated 'online' dataset that includes data from 52B language models. It contains 22k helpfulness comparisons and no red-teaming data. |
| [Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) |                                                              | Instruction-Tuning-with-GPT-4 | 52k    | English | Ranked responses (Note: Data is evaluated by `GPT-4` model NOT human) of Alpaca prompts from three models (GPT-4, GPT-3.5 and OPT-IML) by asking GPT-4 to rate the quality. Author believes "GPT-4 is capable of identifying and fixing its own mistakes, and accurately judging the quality of responses" |
| [thu-coai/Safety-Prompts](https://github.com/thu-coai/Safety-Prompts) | [thu-coai/Safety-Prompts](https://huggingface.co/datasets/thu-coai/Safety-Prompts) |           thu-coai            | 100k   | Chinese | 中文安全prompts，用于评测和提升大模型的安全性，将模型的输出与人类的价值观对齐。 |
| [Chatgpt-Comparison-Detection project](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |                               | 24.3K  | English | Human ChatGPT Comparison Corpus, 60k human answers and 27K ChatGPT answers for around 24K questions. |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | OpenBMB (Tsinghua)            | 64k    | English | Multi-aspect feedback (instruction-following, truthfulness, honesty, helpfulness) for reward/DPO training. |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | NVIDIA                        | 37k    | English | Multi-attribute helpfulness dataset (helpfulness, correctness, coherence, complexity, verbosity). |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | NVIDIA                        | 10k    | English | High-quality preference dataset for reward models; SOTA on RewardBench at release. |
| [HelpSteer2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2-Preference) | [nvidia/HelpSteer2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2-Preference) | NVIDIA                        | 10k    | English | Preference annotations with human-written justifications for BT vs regression reward modeling. |
| [HelpSteer3-Preference](https://huggingface.co/datasets/nvidia/HelpSteer3-Preference) | [nvidia/HelpSteer3-Preference](https://huggingface.co/datasets/nvidia/HelpSteer3-Preference) | NVIDIA                        | 40k+   | English | Diverse open human-annotated preference data across STEM, coding, and multilingual tasks. |
| [Nemotron-Cascade-2-RL-Data](https://huggingface.co/collections/nvidia/nemotron-cascade-2) | [nvidia/Nemotron-Cascade-2-RL-Data](https://huggingface.co/collections/nvidia/nemotron-cascade-2) | NVIDIA | multi-subset | English | RL datasets for Nemotron-Cascade-2 (math, code, reasoning, agentic, instruction following). |
| [TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove) | [open-thoughts/TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove) | OpenThoughts | 750K+ | English | Agentic task specifications in Harbor format for RL and evaluation; companion to AgentTrove. |
| [SYNTHETIC-2-RL](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-RL) | [PrimeIntellect/SYNTHETIC-2-RL](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-RL) | PrimeIntellect | 156k | English | RL subset of SYNTHETIC-2 with difficulty annotations from Qwen3 and DeepSeek-R1-0528. |
| [UltraInteract_preference](https://huggingface.co/datasets/openbmb/UltraInteract_preference) | [openbmb/UltraInteract_preference](https://huggingface.co/datasets/openbmb/UltraInteract_preference) | OpenBMB (Tsinghua) | ~219k | English | Preference trees for multi-turn reasoning and tool use (correct/incorrect trajectories and actions). |
| [RLSTACK](https://huggingface.co/datasets/H-D-T/RLSTACK) | [H-D-T/RLSTACK](https://huggingface.co/datasets/H-D-T/RLSTACK) | Hive-Digital-Technologies | 868k | English | DPO-style preference pairs from Stack Exchange dumps (highest- vs lowest-rated answers). |

<a id="rlhf-dataset-details"></a>
### RLHF Dataset Details

### [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- Summary: This RLHF dataset is an iterated 'online' dataset that includes data from 52B language models. It contains 22k helpfulness comparisons and no red-teaming data.
- Data generation model: `Anthropic RL-CAI 52B`
- paper: [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- Cost: N/A

### [HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)

- Summary: This dataset contains questions and answers from the Stack Overflow Data Dump for the purpose of preference model training.
- Data generation model: N/A
- paper: [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
- Cost: N/A

### [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)

- Summary: Each example is a Reddit post with a question/instruction and a pair of top-level comments for that post, where one comment is more preferred by Reddit users (collectively).
- Data generation model: N/A
- paper: N/A
- Cost: N/A

### [Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

- Summary: Ranked responses (Note: Data is evaluated by `GPT-4` model NOT human) of Alpaca prompts from three models (GPT-4, GPT-3.5 and OPT-IML) by asking GPT-4 to rate the quality. Author believes "GPT-4 is capable of identifying and fixing its own mistakes, and accurately judging the quality of responses"
- Data generation model: `GPT-4`
- paper: [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)
- Cost: N/A
- Related:
    -[(tatsu-lab/Alpaca)|52K|EN|MT|SI](https://github.com/tatsu-lab/stanford_alpaca)


### Natural Instruction / Super-Natural Instruction

- [Paper/Project link](https://aclanthology.org/2022.acl-long.244.pdf)
- [Dataset link](https://instructions.apps.allenai.org/)

Allen AI is the first organization to try Instruction as a prompt and fine-tune LLMs. In the Natural Instruction paper, you can basically understand the labeling ideas of the instruction.

In its proposed dataset, 61 and different NLP tasks are included.

Super-Natural Instruction is a super-intensive version of Natural Instruction, which contains more than 1,600 different NLP tasks, and there are more than 76 different types of NLP tasks (such as: classification, extraction, sequence labeling).

### [BigScience/P3](https://huggingface.co/datasets/bigscience/P3)

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

### [Unnatural Instruction](https://github.com/orhonovich/unnatural-instructions)

- [Paper/Project Link](https://arxiv.org/pdf/2212.09689.pdf)
- [Dataset Link](https://github.com/orhonovich/unnatural-instructions)

Using LLMs to independently generate instruction data is an active direction in the field of instruction-tuning.

Unnatural Instruction uses GPT3 (text-davinci-002) to generate 64k instruction prompt data. And use the same model to rewrite the 64k prompt, and finally get 240k instruction data.

The paper shows that the prompts generated by LLMs in Instruct-Tuning show good results, even surpassing models such as T0 that are fine-tuned on P3 and other data.

### [Self-Instruct](https://github.com/yizhongw/self-instruct)

- [Paper/Project Link](https://arxiv.org/pdf/2212.10560.pdf)
- [Dataset Link](https://github.com/yizhongw/self-instruct)

Self-Instruct is also the idea of using LLMs to generate prompts for instruction-tuning. However, a more fine-grained generation process is used.

Concepts such as Task pool and Quality filtering were introduced to partially alleviate the noise problem of self-intrauct type data.

### [UnifiedSKG - HKU](https://unifiedskg.com/)

- [Paper/Project Link](https://arxiv.org/pdf/2201.05966.pdf)

- [DataSet Link](https://unifiedskg.com/)

UnifiedSKG has added knowledge grounding in the Text-to-Text framework, that is, in the prompt-output framework, it has added structured data for assistance.

As an example, some NLP tasks rely heavily on structured knowledge bases/databases. The idea of UnifiedSKG is to serialize the required database and embed it into the prompt. As shown below.

UnifiedSKG represents a direction in the field of LLMs that attempts to use structured knowledge to enhance performance.

### [Google/Flan Collection](https://github.com/google-research/FLAN/tree/main/flan/v2)

- [Paper/Project Link](https://arxiv.org/pdf/2301.13688.pdf)
- [Dataset Link](https://github.com/google-research/FLAN/tree/main/flan/v2)

In this project, Google merged its own Flan 2021 data with some open source instruction data (P3, super-natural instruction, etc.).

In Flan Collection's paper, Google also summarizes some key points in Flan series model training/reasoning, which may have good reference value.

The Flan Collection compiles datasets from Flan 2021, P3, Super-Natural Instructions, along with dozens more datasets into one place, formats them into a mix of zero-shot, few-shot and chain-of-thought templates

- 
### InstructDial

- [Paper/Project Link](https://arxiv.org/pdf/2205.12673.pdf)
- [Dataset Link](https://github.com/prakharguptaz/Instructdial/tree/main/datasets)

InstructDial is an attempt to fine-tune instructions on a specific task type. Experimental results show that after fine-tuning on dialogue instruction data, the model performs better on dialogue tasks than on very large-scale task sets.


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

### [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [Paper/Project Link](https://arxiv.org/abs/2310.01377)
- [Dataset Link](https://huggingface.co/datasets/openbmb/UltraFeedback)

A large-scale, multi-aspect feedback dataset for training reward models and DPO. It contains ~64k prompts with multiple ranked responses annotated for instruction-following, truthfulness, honesty and helpfulness.

### [HelpSteer / HelpSteer2 / HelpSteer2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [Paper/Project Link](https://arxiv.org/abs/2311.09528)
- [Dataset Link](https://huggingface.co/datasets/nvidia/HelpSteer)
- [Dataset Link 2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [Dataset Link 2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2-Preference)

NVIDIA's family of multi-attribute helpfulness datasets. HelpSteer provides fine-grained ratings across helpfulness, correctness, coherence, complexity and verbosity. HelpSteer2 adds preference pairs for reward model training; HelpSteer2-Preference includes human-written justifications comparing Bradley-Terry and regression reward modeling. HelpSteer3-Preference expands to 40k+ diverse real-world applications including STEM, coding, and multilingual tasks.

### [Nemotron-Cascade-2-RL-Data](https://huggingface.co/collections/nvidia/nemotron-cascade-2)
- [Collection Link](https://huggingface.co/collections/nvidia/nemotron-cascade-2)
- [Paper](https://arxiv.org/abs/2603.19220)
- Summary: Reinforcement-learning datasets used to train Nemotron-Cascade-2-30B-A3B. Includes domain-wise RL data for math, code, reasoning, instruction following, long-context, agentic/SWE, and RLHF-style preference pairs. Released alongside the SFT data and model checkpoints.
- Data generation model: `Nemotron-Cascade-2 pipeline`
- License: NVIDIA Open Model License

### [TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove)
- [Dataset Link](https://huggingface.co/datasets/open-thoughts/TaskTrove)
- [GitHub](https://github.com/open-thoughts/OpenThoughts-Agent)
- Summary: 750K+ agentic task specifications in the Harbor format, designed for RL and evaluation of agentic coding/computer-use models. Companion to AgentTrove; tasks can be deployed via Harbor to update model behavior.
- Data generation model: `COL`
- License: Apache 2.0

### [SYNTHETIC-2-RL](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-RL)
- [Dataset Link](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-RL)
- [Blog](https://www.primeintellect.ai/blog/synthetic-2-release)
- Summary: 156K RL-focused problems from the SYNTHETIC-2 collection with difficulty/reward annotations from Qwen3-32B, Qwen3-4B, DeepSeek-R1-0528-Qwen3-8B, and DeepSeek-R1-0528. Each row includes verification info suitable for rule-based reward training.
- Data generation model: `DeepSeek-R1-0528`, `Qwen3`

### [UltraInteract_preference](https://huggingface.co/datasets/openbmb/UltraInteract_preference)
- [Dataset Link](https://huggingface.co/datasets/openbmb/UltraInteract_preference)
- [Paper](https://arxiv.org/abs/2404.02078)
- Summary: ~219K preference pairs (and up to 286K correct/incorrect nodes) organized as preference trees for multi-turn reasoning and tool-use trajectories. Supports both reward-model and DPO-style training.
- Data generation model: `GPT-4`

### [RLSTACK](https://huggingface.co/datasets/H-D-T/RLSTACK)
- [Dataset Link](https://huggingface.co/datasets/H-D-T/RLSTACK)
- Summary: 868K DPO-style preference pairs built from Stack Exchange dumps across 53 communities. Each example contains a prompt and paired `chosen` (highest-rated) / `rejected` (lowest-rated) answers.
- Data generation model: N/A
- License: CC BY 4.0

<a id="dpo-datasets"></a>
# DPO Datasets

<a id="dpo-statistics"></a>
### DPO Statistics

| Project                                                      | Links                                                        | Org                           | Nums     | Lang    | Summary                                                      |
| :----------------------------------------------------------- | :----------------------------------------------------------: | :---------------------------: | :------- | :-----: | :----------------------------------------------------------- |
| [Magpie-Pro-DPO](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-DPO-100K-v0.1) | [Magpie-Align/Magpie-Pro-DPO-100K-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-DPO-100K-v0.1) | Magpie-Align                  | 100k     | English | Preference pairs synthesized from aligned LLMs for DPO training. |
| [Orca-DPO-Pairs](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs) | [HuggingFaceH4/orca_dpo_pairs](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs) \|[Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) \|[argilla/distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) | HuggingFaceH4 / Intel / Argilla | 12k / 6k | English | DPO preference pairs from GPT-4 Orca reasoning traces; Argilla cleaned version ~6k. |
| [UltraFeedback-Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | HuggingFaceH4                 | 61k-230k | English | Binarized UltraFeedback into chosen/rejected pairs for DPO; decontaminated variants available. |
| [DPO-Mix-7K](https://huggingface.co/datasets/argilla/dpo-mix-7k) | [argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k) | Argilla                       | 7.5k     | English | Curated cocktail of Capybara, Intel Orca, and UltraFeedback high-quality DPO pairs. |
| [Llama3-UltraFeedback-ArmoRM](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) | [princeton-nlp/llama3-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) | Princeton NLP                 | 60k      | English | 5 Llama-3-SFT responses per prompt ranked by ArmoRM for advanced preference research. |
| [ChatML-DPO-Pairs](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs) | [mlabonne/chatml-dpo-pairs](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs) | Maxime Labonne                | 12k      | English | ChatML-formatted Intel Orca DPO pairs (ChatGPT chosen vs Llama-2 rejected). |
| [Magpie-Air-DPO](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1) | [Magpie-Align/Magpie-Air-DPO-100K-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1) | Magpie-Align                  | 100k     | English | Synthetic DPO preference pairs from Llama-3-8B-Instruct (Magpie-Air). |
| [Tulu 3 Preference](https://github.com/allenai/open-instruct) | [allenai/tulu-3-pref-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following) \|[allenai/tulu-3-pref-mixture](https://huggingface.co/datasets/allenai/tulu-3-pref-mixture) | Ai2                           | 20k-200k | English | Preference data used in Tulu 3 post-training pipeline. |
| [RLSTACK](https://huggingface.co/datasets/H-D-T/RLSTACK) | [H-D-T/RLSTACK](https://huggingface.co/datasets/H-D-T/RLSTACK) | Hive-Digital-Technologies | 868k | English | Stack Exchange DPO-style preference pairs (chosen/rejected answers). |
| [UltraInteract_preference](https://huggingface.co/datasets/openbmb/UltraInteract_preference) | [openbmb/UltraInteract_preference](https://huggingface.co/datasets/openbmb/UltraInteract_preference) | OpenBMB (Tsinghua) | ~219k | English | Preference trees for multi-turn reasoning and tool-use trajectories. |
| [Ling-Coder-DPO](https://huggingface.co/datasets/inclusionAI/Ling-Coder-DPO) | [inclusionAI/Ling-Coder-DPO](https://huggingface.co/datasets/inclusionAI/Ling-Coder-DPO) | inclusionAI / Codefuse | 253k | English | Code-specific DPO pairs curated from code_contests using test-pass, PPL, and reward-model signals. |
| [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | ByteDance / Tsinghua SIA | 1.79M | English | Competition-level math problems with reward metadata for RL/DPO training. |

<a id="dpo-dataset-details"></a>
### DPO Dataset Details

### [Orca-DPO-Pairs](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs)
- [Alternative Source](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [Cleaned Version](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)

~12k DPO preference pairs derived from GPT-4 augmented Orca/FLAN reasoning traces. Intel released a widely used variant; Argilla later cleaned it to ~6k high-quality pairs, swapping/rejecting mislabeled examples.

### [UltraFeedback-Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

61k–230k preference pairs obtained by binarizing UltraFeedback scores into chosen/rejected responses. A standard DPO dataset used in Zephyr, Tulu 3, and many open alignment recipes.

### [DPO-Mix-7K](https://huggingface.co/datasets/argilla/dpo-mix-7k)
- [Dataset Link](https://huggingface.co/datasets/argilla/dpo-mix-7k)

A small, curated cocktail (~7.5k pairs) mixing high-scoring chosen responses from Capybara, Intel Orca, and UltraFeedback. Designed for efficient, high-quality DPO training.

### [Llama3-UltraFeedback-ArmoRM](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm)
- [Dataset Link](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm)

~60k prompts each with five Llama-3-SFT responses ranked by the ArmoRM reward model. Supports advanced preference optimization research beyond binary pairs.

### [ChatML-DPO-Pairs](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs)
- [Dataset Link](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs)

~12k ChatML-formatted preference pairs based on Intel/orca_dpo_pairs, with ChatGPT responses as chosen and Llama-2-13b-chat responses as rejected.

### [Magpie-Air-DPO](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1)
- [Dataset Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1)

~100k synthetic DPO preference pairs generated from Llama-3-8B-Instruct using the Magpie-Air pipeline.

### [Tulu 3 Preference](https://github.com/allenai/open-instruct)
- [Dataset Link](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)
- [Preference Mixture](https://huggingface.co/datasets/allenai/tulu-3-pref-mixture)

Ai2's Tulu 3 preference data (~20k–200k) including persona-based instruction-following preferences and a full preference mixture for open post-training.

<a id="rlstack"></a>
### RLSTACK
> Cross-reference: see the full [RLSTACK](#rlstack) entry in the RLHF / Preference Datasets section. It is also used as a DPO-style preference dataset.

<a id="ultrainteract_preference"></a>
### UltraInteract_preference
> Cross-reference: see the full [UltraInteract_preference](#ultrainteract_preference) entry in the RLHF / Preference Datasets section. It is also used for DPO training.

### [Ling-Coder-DPO](https://huggingface.co/datasets/inclusionAI/Ling-Coder-DPO)
- [Dataset Link](https://huggingface.co/datasets/inclusionAI/Ling-Coder-DPO)
- [Paper](https://arxiv.org/abs/2503.17793)

253K code-specific preference pairs derived from code_contests. Positive/negative samples are selected using code test-case pass rates, perplexity distribution, and reward model scores. Used to train Ling-Coder-Lite.

### [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- [Dataset Link](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- [Collection](https://huggingface.co/collections/BytedTsinghua-SIA/dapo)

1.79M competition-level math problems with reward-model metadata (`ground_truth`, evaluation style). Intended for RL and preference optimization such as GRPO/DPO; part of the DAPO collection.

<a id="datasets-without-license-information"></a>
<a id="modelscope-chinese-datasets"></a>
# ModelScope Chinese Datasets
A curated set of Chinese instruction-tuning and agent datasets available on [ModelScope](https://www.modelscope.cn/). These are also listed in the Multilingual & Chinese tables above; this section provides quick access to ModelScope links.

| Dataset | ModelScope Link | Size | Description |
| :------ | :-------------- | :--- | :---------- |
| **COIG-CQIA** | [m-a-p/COIG-CQIA](https://modelscope.cn/datasets/m-a-p/COIG-CQIA) | 48k | High-quality Chinese instructions following the "Quality is All You Need" philosophy. |
| **MSAgent / MSAgent-Bench** | [iic/MSAgent-Bench](https://modelscope.cn/datasets/iic/MSAgent-Bench) | 600k+ | Chinese/English tool-use and agent training data. |
| **DeepCtrl-SFT** | [deepctrl/deepctrl-sft-data](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data) | 12M | Safe Chinese/English general-domain SFT data (~3B tokens). |
| **Infinity Instruct** | [AI-ModelScope/Infinity-Instruct](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct) | 7M+ | Large-scale Chinese/English instruction collection by BAAI/FlagOpen. |
| **Alpaca-GPT4 Chinese** | [AI-ModelScope/alpaca-gpt4-data-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh) | 52k | Chinese Alpaca instruction data generated by GPT-4. |
| **HC3-Chinese** | [simpleai/HC3-Chinese](https://modelscope.cn/datasets/simpleai/HC3-Chinese) | 39,781 | Chinese human vs ChatGPT response comparison corpus. |
| **OpenHermes 2.5** | [swift/OpenHermes-2.5](https://modelscope.cn/datasets/swift/OpenHermes-2.5) | ~1M | High-quality general instruction mixture. |
| **Magpie-Qwen2-Pro-200K-Chinese** | [HuggingFace](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese) | 200k | Chinese alignment synthesis data based on Qwen2. |
| **Chinese-DeepSeek-R1-Distill-110k-SFT** | [liucong/Chinese-DeepSeek-R1-Distill-data-110k](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k) | 110k | Chinese reasoning SFT data distilled from DeepSeek-R1-671B. |
| **Seq-Monkey** | [ddzhu123/seq-monkey](https://modelscope.cn/datasets/ddzhu123/seq-monkey) | 10B tokens | Mobvoi's Chinese general pre-training corpus. |
| **smoltalk-chinese** | [openscsg/smoltalk-chinese](https://modelscope.cn/datasets/openscsg/smoltalk-chinese) | 700k+ | Chinese general instruction-tuning data. |
| **OpenThoughts3-1.2M** | [open-thoughts/OpenThoughts3-1.2M](https://modelscope.cn/datasets/open-thoughts/OpenThoughts3-1.2M) | 1.2M | OpenThoughts 3 reasoning dataset (math / code / science). |

> Tip: You can use these datasets directly with the [ms-swift](https://github.com/modelscope/ms-swift) framework.

# Datasets without license information

 ## [alespalla/chatbot_instruction_prompts](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts)

 - Summary: A compilation of `tatsu-lab/alpaca` ,`Dahoas/instruct-human-assistant-prompt` ,`allenai/prosocial-dialog`
 - Data generation model: N/A
 - paper: N/A
 - Cost: N/A

<a id="contributing"></a>
# Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](#contribution-template) for instructions in contribution.

<a id="contribution-template"></a>
## Contribution Template
Append the new project at the end of file:

```markdown
[{Project-name}/{Dataset-name}](https://github.com/link/to/project)

- [paper/project link](link)
- [dataset link](link)
- Related work: (if applicable)

Some introductions ...
```

<a id="license"></a>
# License

`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.

<a id="reference"></a>
# Reference
- https://github.com/Zjh-819/LLMDataHub
- https://github.com/raunak-agarwal/instruction-datasets
- https://github.com/zhilizju/Awesome-instruction-tuning
- https://github.com/RenzeLou/awesome-instruction-learning
- https://github.com/neuml/txtinstruct
- https://github.com/modelscope/ms-swift
- https://github.com/ymcui/Chinese-LLaMA-Alpaca
- https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
- https://github.com/magpie-align/magpie
- https://github.com/FlagOpen/Infinity-Instruct
- https://github.com/hkust-nlp/deita
- https://github.com/hkust-nlp/dart-math
- https://github.com/GAIR-NLP/LIMO
- https://github.com/tencent-ailab/persona-hub
- https://github.com/allenai/open-instruct
- https://github.com/OpenCoder-llm/OpenCoder-llm
- https://github.com/open-thoughts/open-thoughts
- https://github.com/open-thoughts/OpenThoughts-Agent
- https://github.com/huggingface/open-r1
- https://github.com/PrimeIntellect-ai/genesys
- https://github.com/bespokelabsai/curator
- https://github.com/NVIDIA-NeMo/Nemotron

[NEUIR/LongMab-PO](https://github.com/NEUIR/LongMab-PO)

- [paper/project link](https://aclanthology.org/2026.acl-long.161/)
- [dataset link](https://drive.google.com/drive/folders/1uK3-yFqgnpmjvc_M6Q89WrZYPcZ_JJlG)

Synthetic preference pairs for long-context DPO, with separate releases for Llama 3 and Qwen 2. LongMab-PO treats context chunks as multi-armed-bandit arms and uses reward feedback to sample diverse responses and construct preference pairs.
