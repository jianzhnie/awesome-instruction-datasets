
<div align="center">

# Awesome Instruction Datasets 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
</div>

<div align="center">

[中文](README_zh.md) | English
</div>

# Contents
- [Awesome Prompt datasets](#awesome-prompt-datasets)
- [Contents](#contents)
- [Introduction](#introduction)
- [Prompt Datasets](#prompt-datasets)
  - [Statistics](#statistics)
- [RLHF Datasets](#rlhf-datasets)
  - [Statistics](#statistics-1)
- [The template](#the-template)
- [The Prompt Datasets List](#the-prompt-datasets-list)
  - [Alpaca -Stanford](#alpaca--stanford)
  - [Instruction in the Wild](#instruction-in-the-wild)
  - [JosephusCheung/GuanacoDataset](#josephuscheungguanacodataset)
  - [Stanford Human Preferences Dataset (SHP)](#stanford-human-preferences-dataset-shp)
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
  - [BELLE/data/1.5M](#belledata15m)
  - [alpaca\_chinese\_dataset](#alpaca_chinese_dataset)
  - [Med-ChatGLM/data](#med-chatglmdata)
  - [pCLUE](#pclue)
  - [COIG](#coig)
- [The RLHF Datasets List](#the-rlhf-datasets-list)
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
- [Contributing](#contributing)
- [License](#license)


# Introduction
"Welcome to 'awesome-prompt-datasets', a comprehensive collection of high-quality open-source instruction tuning datasets to train chat-based LLMs (ChatGPT,LLaMA,Alpaca)。

Instruction Tuning / Reinforcement Learning from Human Feedback (RLHF) Dataset is a key component of instruction-following LLMs such as ChatGPT. This repo is dedicated to providing a comprehensive list of datasets used for instruction tuning in various LLMs, making it easier for researchers and developers to access and utilize these resources.

With 'awesome-prompt-dataset', you can accelerate your research and development in NLP and unlock new opportunities for innovation. Let's explore the possibilities together!"

# Prompt Datasets

Referring to [this](https://github.com/yaodongC/awesome-instruction-dataset) ([@yaodongC](https://github.com/yaodongC)), we labeled each collected dataset according to the following rules:

**(Lang)Lingual-Tags**:

- EN: Instruction datasets in English
- CN: Instruction datasets in Chinese
- ML: [Multi-lingual] Instruction datasets in multiple languages

**(Task)Task-Tags**:

- MT: [Multi-task] Datasets containing multiple tasks
- TS: [Task-specific] Datasets tailored for specific tasks

**(Gen)Generation-method**:

- HG: [Human Generated Dataset] Datasets created by humans
- SI: [Self-Instruct] Datasets generated using self-instruct methods
- MIX: [Mixed Dataset] Dataset contains both human and machine generated data
- COL: [Collection of Dataset] Dataset made from a collection of other datasets

## Statistics

| Project                                                      |                           Datasets                           | Org                        | Nums      | Lang  | Task  | Gen  | Type                                                         | Src                                                          | Url                                                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: | -------------------------- | :-------- | :---- | :---- | :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Chain of Thought](https://github.com/google-research/FLAN)  | [cot_data](https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data) \|[few_shot_data](https://github.com/google-research/FLAN/tree/main/flan/v2/niv2_few_shot_data) | Google                     | 74771     | EN/CN | MT    | HG   | instruct with cot reasoning                                  | annotating CoT on existing data                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chain-of-Thought) |
| [GPT4all](https://github.com/nomic-ai/gpt4all)               | [nomic-ai/gpt4all-j-prompt-generations](https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations) | nomic-ai                   | 806199    | EN    | MT    | COL  | code, storys and dialogs                                     | distillation from GPT-3.5-turbo                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all) |
| [GPTeacher](https://github.com/teknium1/GPTeacher)           | [GPT-4 General-Instruct ](https://github.com/teknium1/GPTeacher/tree/main/Instruct)\|[Roleplay-Instruct](https://github.com/teknium1/GPTeacher/tree/main/Roleplay) \|[Code-Instruct ](https://github.com/teknium1/GPTeacher/tree/main/Codegen)\| [Toolformer](https://github.com/teknium1/GPTeacher/tree/main/Toolformer) | teknium1                   | 29013     | EN    | MT    | SI   | general, roleplay, toolformer                                | GPT-4 & toolformer                                           | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPTeacher) |
| [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | [JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | JosephusCheung             | 534610    | ML    | MT    | SI   | various linguistic tasks                                     | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Guanaco) |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)    | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Hello-SimpleAI \| 万得资讯 | 37175     | EN/CN | TS    | MIX  | dialogue evaluation                                          | human or ChatGPT                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/HC3) |
| [HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | [Hello-SimpleAI/HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) | Hello-SimpleAI\|万得资讯   | 13k       | CN    | TS    | MIX  | dialogue evaluation                                          | human or ChatGPT                                             |                                                              |
| [alpaca](https://github.com/tatsu-lab/stanford_alpaca)       | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | tatsu-lab                  | 52002     | EN    | MT    | SI   | general instruct                                             | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca) |
| [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | yahma                      | 52k       | EN    | MT    | SI   | general instruct                                             | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca) |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [alpaca_data_zh_51k](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/data/alpaca_data_zh_51k.json) | ymcui(讯飞)                | 51k       | CN    | MT    | SI   | general instruct                                             | text-davinci-003                                             |                                                              |
| [Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM)  骆驼 | [trans_chinese_alpaca_data](https://github.com/LC1332/Luotuo-Chinese-LLM/blob/main/data/trans_chinese_alpaca_data.json) | LC1332(商汤)               | 52k       | CN    | MT    | SI   | general instruct                                             | text-davinci-003                                             |                                                              |
| [Natural Instructions](https://github.com/allenai/natural-instructions) | [Allen AI 61 task](https://instructions.apps.allenai.org/#:~:text=Download%20Natural%2DInstructions%20%2D%20v1.1)\|[1.5k task](https://instructions.apps.allenai.org/#:~:text=Natural%2DInstructions%20%2D%20v2-,.,-x) | Allen AI                   | 5040134   | ML    | MT    | COL  | diverse nlp tasks                                            | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Natural-Instructions) |
| [belle_cn](https://huggingface.co/BelleGroup)                | [BelleGroup/train_1M_CN](https://huggingface.co/datasets/bellegroup/train_1M_CN) \|[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/bellegroup/train_0.5M_CN) | BelleGroup(链家)           | 1079517   | CN    | TS/MT | SI   | general, mathematical reasoning, dialogue                    | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/belle_cn) |
| [instinwild](https://github.com/XueFuzhao/InstructionWild)   | [instinwild_ch](https://github.com/XueFuzhao/InstructionWild/tree/main/data) \| [instinwild_en](https://github.com/XueFuzhao/InstructionWild/tree/main/data) |                            | 52191     | EN/CN | MT    | SI   | generation, open-qa, mind-storm                              | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instinwild) |
| [华驼(HuaTuo)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | [中文医学知识](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/data/llama_data.json) \|[肝癌](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/data-literature/liver_cancer.json) | SCIR-HI(哈工大)            | 8K        | CN    | TS    | SI   | 公开和自建的中文医学知识库                                   | GPT3.5                                                       |                                                              |
| [prosocial dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | allenai                    | 165681    | EN    | TS    | MIX  | dialogue                                                     | GPT-3 rewrites questions + humans feedback manually          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/prosocial-dialog) |
| [finance_en](https://huggingface.co/datasets/gbharti/finance-alpaca) | [gbharti/finance-alpaca](https://huggingface.co/datasets/allenai/prosocial-dialog) |                            | 68912     | EN    | TS    | COL  | financial related qa                                         | GPT3.5                                                       | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/) |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)        | [bigscience/xP3](https://huggingface.co/datasets/bigscience/xP3) | bigscience                 | 78883588  | ML    | MT    | COL  | a collection of prompts & datasets across 46 of languages & 16 NLP tasks | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/xP3) |
| [firefly](https://github.com/yangjianxin1/Firefly)           | [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) |                            | 1649398   | CN    | MT    | COL  | 23 nlp tasks                                                 | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/firefly) |
| [instruct](https://huggingface.co/datasets/swype/instruct)   | [swype/instruct](https://huggingface.co/datasets/swype/instruct) |                            | 888969    | EN    | MT    | COL  | augmented of GPT4All, Alpaca, open-source Meta datasets      | augmentation performed using the advanced NLP tools provided by AllenAI | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instruct) |
| [Code Alpaca](https://github.com/sahil280114/codealpaca)     | [sahil280114/codealpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) |                            | 20022     | EN    | TS    | SI   | code generation, editing, optimization                       | text-davinci-003                                             | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/CodeAlpaca) |
| [Alpaca_GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | [alpaca_gpt4_data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)\|[alpaca_gpt4_data_zh](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json) \|[comparison_data_v2](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/comparison_data_v2.json) | 微软                       | 52002     | EN/CN | MT    | SI   | general instruct                                             | generated by GPT-4 using Alpaca                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpacaGPT4) |
| [webGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) | [openai/webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons) | openai                     | 18994     | EN    | TS    | MIX  | information retrieval (IR) QA                                | fine-tuned GPT-3, each instruction has two outputs, select better one | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/webGPT) |
| [dolly 2.0](https://github.com/databrickslabs/dolly)         | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | databricks                 | 15015     | EN    | TS    | HG   | closed QA , summarization and etc, Wikipedia as references   | human annotated                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/dolly) |
| [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry) | [mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) | mosaicml                   | 59.3K     | EN    | TS    | HG   | This dataset is a combination of [Databrick's dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset and a filtered subset of [Anthropic's HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf). | human annotated                                              |                                                              |
| [baize](https://github.com/project-baize/baize-chatbot) 白泽 | [alpaca_chat_data.json](https://github.com/project-baize/baize-chatbot/tree/main/data) \|[medical_chat_data.json](https://github.com/project-baize/baize-chatbot/blob/main/data/medical_chat_data.json) \| [quora_chat_data.json](https://github.com/project-baize/baize-chatbot/blob/main/data/quora_chat_data.json) \|[stackoverflow_chat_data.json](https://github.com/project-baize/baize-chatbot/blob/main/data/stackoverflow_chat_data.json) | project-baize              | 653699    | EN    | MT    | COL  | a collection from Alpaca, Quora, StackOverFlow and MedQuAD questions | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/baize) |
| [hh-rlhf](https://github.com/anthropics/hh-rlhf)             | [Anthropic/hh-rlhf](https://huggingface.co/datasets/anthropic/hh-rlhf) | Anthropic                  | 284517    | EN    | TS    | MIX  | dialogue                                                     | dialog between human and RLHF models                         | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/hh-rlhf) |
| [OIG(part)](https://laion.ai/blog/oig-dataset/)              |    [laion/OIG](https://huggingface.co/datasets/laion/oig)    | laion                      | 49237     | EN    | MT    | COL  | created from various tasks, such as question and answering   | using data augmentation, human annotated datasets collection | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/OIG) |
| [GAOKAO](https://github.com/OpenLMLab/GAOKAO-Bench)          | [Fill-in-the-blank_Questions](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data/Fill-in-the-blank_Questions) \| [Multiple-choice_Questions](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data/Multiple-choice_Questions) \| [Open-ended_Questions](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data/Open-ended_Questions) | OpenLMLab                  | 2785      | CN    | MT    | COL  | Multiple-choice, Fill-in-the-blank and Open-ended questions from examination | human annotated                                              | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GAOKAO) |
| [camel](https://github.com/lightaime/camel) \| 骆驼          | [camel-ai/code](https://huggingface.co/datasets/camel-ai/ai_society)\|[camel-ai/biology](https://huggingface.co/datasets/camel-ai/biology) \|[camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics) \|[camel-ai/chemistry](https://huggingface.co/datasets/camel-ai/chemistry) \|[camel-ai/math](https://huggingface.co/datasets/camel-ai/math) | camel-ai                   | 760620    | EN    | MT    | SI   | Role-Playing conversations in AI Society, Code, Math, Physics, Chemistry, Biolog | gpt-3.5-turbo                                                | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/camel) |
| [FLAN-Muffin](https://huggingface.co/datasets/Muennighoff/flan) | [Muennighoff/flan](https://huggingface.co/datasets/Muennighoff/flan) |                            | 1764800   | EN    | MT    | COL  | 60 nlp tasks                                                 | human annotated datasets collection                          | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/FLAN-Muffin) |
| [COIG](https://huggingface.co/datasets/BAAI/COIG)            |      [COIG](https://huggingface.co/datasets/BAAI/COIG)       | BAAI\|智源                 | 298428    | CN    | MT    | COL  | collect fron Exam, Translated, Human Value Alignment Instructions and Counterfactural Correction Multi-round Chat | using automatic tool and manual verification                 | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/COIG) |
| [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)        | [gpt4tools_71k.json](https://drive.google.com/file/d/1JKIT-Or1of7TJuWvmrJpPoOx0cLdcWry/view?usp=share_link) | StevenGrove                | 71446     | EN    | MT    | SI   | a collection of tool-related instructions                    | gpt-3.5-turbo                                                | [download](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/gpt4tools) |
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
| [WizardLM](https://github.com/nlpxucan/WizardLM)             | [victor123/evol_instruct_70k](https://huggingface.co/datasets/victor123/evol_instruct_70k) | WizardLM                   | 70k       | EN    | MT    |      |                                                              |                                                              |                                                              |
|                                                              |                                                              |                            |           |       |       |      |                                                              |                                                              |                                                              |

# RLHF Datasets

## Statistics

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

# Open ChatLLMs

| Release    | Model_name                                                   | Base          | Model_Size | Datasets                                                     | Number of Instances | Language    |
| ---------- | ------------------------------------------------------------ | ------------- | ---------- | ------------------------------------------------------------ | ------------------- | ----------- |
| 2022-12    | GPT-3 Self Inst.                                             | GPT-3         | 175B       | Self-Instruct                                                | 82 k                | En          |
| 2023-03-03 | [alpaca](https://github.com/tatsu-lab/stanford_alpaca)       | LLaMA         | 7B         | [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) | 52 k                | En          |
| 2023-03-19 | [alpaca-lora](https://github.com/tloen/alpaca-lora/commits/main) | LLaMA         | 7B 13B 30B | [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)、[alpaca_data_cleaned](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_cleaned.json) | 52 k                | En          |
| 2023-03-23 | [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)   | LLaMA         | 7B 13B     | [BELLE](https://github.com/LianjiaTech/BELLE)、[GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | 1M                  | Zh          |
| 2023-03-24 | [Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)        | LLaMA         | 7B         | [dataset](https://github.com/PhoebusSi/Alpaca-CoT#statistics) | ----                | En Zh       |
| 2023-03-25 | [dolly](https://github.com/databrickslabs/dolly)             | dolly         | 6B         | [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) | 52 k                | En          |
| 2023-03-25 | [guanaco](https://huggingface.co/KBlueLeaf/guanaco-7B-leh)   | LLaMA         | 7B         | [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | 534 k               | En Zh Ja De |
| 2023-03-28 | [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | LLaMA         | 7B         | [alpaca_data_zh](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/data)、[pCLUE](https://github.com/CLUEbenchmark/pCLUE)、[translation2019zh](https://github.com/brightmart/nlp_chinese_corpus#5%E7%BF%BB%E8%AF%91%E8%AF%AD%E6%96%99translation2019zh)、[alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)、Self-Instruct | 2M                  | Zh          |
| 2023-03-29 | [ColossalChat](https://github.com/hpcaitech/ColossalAI)      | LLaMA         | 7B 13B     | [InstructionWild](https://github.com/XueFuzhao/InstructionWild) | 104 k               | En Zh       |
| 2023-03-31 | [Luotuo](https://github.com/LC1332/Luotuo-Chinese-LLM)       | LLaMA ChatGLM | 7B 6B      | [trans_chinese_alpaca_data](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json) | 52k                 | Zh          |
| 2023-03-31 | [cerebras-lora-alpaca](https://github.com/lxe/cerebras-lora-alpaca) | Cerebras-GPT  | 2.7B       | [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) | 52k                 | En          |

# The template

Append the new project at the end of file
```shell

[{Project-name}/{Dataset-name}]{https://github.com/link/to/project}

- [paper/project link](link)
- [dataset link](link)
- Related work: (if applicable)

Some introductions ...

```

# The Prompt Datasets List

## [Alpaca -Stanford](https://github.com/tatsu-lab/stanford_alpaca)

- [Paper/Project Link](https://github.com/tatsu-lab/stanford_alpaca)
- [Dataset Link](https://github.com/tatsu-lab/stanford_alpaca)
- Data generation model: text-davinci-003
- Cost: $600

The Alpaca of the Stanford release is a fine-tuning model for instruct-tuning based on the Meta Ai LLaMA model.

Alpaca automatically generated 52k instruction data using GPT-3.5 and used it to fine-tune the LLaMA model. Experimental results show that it can reach or even exceed the performance of GPT-3.5 on some tasks.

## [Instruction in the Wild](https://github.com/XueFuzhao/InstructionWild)

- [Paper/Project Link](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat)
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

## COIG

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

# The RLHF Datasets List

## [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- Summary: This RLHF dataset is an iterated 'online' dataset that includes data from 52B language models. It contains 22k helpfulness comparisons and no red-teaming data. 
- Data generation model: `Anthropic RL-CAI 52B`
- paper: [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- Cost: N/A

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

# Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

# License

`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.



## Reference
- https://github.com/Zjh-819/LLMDataHub
- https://github.com/raunak-agarwal/instruction-datasets
- https://github.com/zhilizju/Awesome-instruction-tuning
- https://github.com/RenzeLou/awesome-instruction-learning
- https://github.com/neuml/txtinstruct
