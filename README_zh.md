
<div align="center">

# Awesome Instruction Datasets
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
</div>

<div align="center">

[English](README.md) | 中文
</div>

## 目录

- [简介](#简介)
- [更新说明 (2023–2025)](#更新说明-20232025)
- [快速分类索引](#快速分类索引)
- [SFT 数据集](#sft-数据集)
  - [SFT 数据统计](#sft-数据统计)
  - [SFT 数据集详情](#sft-数据集详情)
- [RLHF / 偏好数据集](#rlhf--偏好数据集)
  - [RLHF 数据统计](#rlhf-数据统计)
  - [RLHF 数据集详情](#rlhf-数据集详情)
- [DPO 偏好优化数据集](#dpo-偏好优化数据集)
  - [DPO 数据统计](#dpo-数据统计)
  - [DPO 数据集详情](#dpo-数据集详情)
- [ModelScope 中文指令微调数据集](#modelscope-中文指令微调数据集)
- [无许可信息的数据集](#无许可信息的数据集)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [参考](#参考)


## 简介

这是一个用于中文指令调整的 AWESOME 数据集合集。

通过使用指令进行微调，以提高 LLM（大型语言模型）的性能成为了一个趋势。随着以数据为中心的 AI 越来越受欢迎，我们需要更高质量的数据集来训练我们的模型。

在这里，你可以找到一些开源的中文指令数据的 AWESOME 列表。


## 更新说明 (2023–2025)

指令微调数据生态发展迅速。本列表在保留经典数据集的基础上，新增了近两年具有代表性的高质量数据集，覆盖通用 SFT、RLHF/偏好、代码/数学推理以及中文资源。特别推荐在 [ModelScope 魔搭社区](#modelscope-中文指令微调数据集) 获取中文数据集。

新增亮点：
- **SFT 数据集**：LIMA、UltraChat、OpenOrca/SlimOrca、OpenHermes 2.5、WizardLM/Evol-Instruct、MetaMathQA、Airoboros、Platypus、No Robots、Magpie、Infinity Instruct、Deita、WildChat、Capybara、Tulu 3 SFT Mix、Smol-Smoltalk、PersonaHub、OpenMathInstruct-1、DART-Math、LIMO、Code-Feedback、FineWeb-Edu、OpenCodeInstruct。
- **RLHF / 偏好数据集**：Anthropic HH-RLHF、Stanford SHP、UltraFeedback、HelpSteer/HelpSteer2/HelpSteer2-Preference、NVIDIA HelpSteer3-Preference。
- **DPO 数据集**：Orca-DPO-Pairs、UltraFeedback-Binarized、DPO-Mix-7K、Llama3-UltraFeedback-ArmoRM、ChatML-DPO-Pairs、Magpie-Air-DPO、Magpie-Pro-DPO、Tulu 3 Preference。
- **代码 / 数学**：MetaMathQA、OpenMathInstruct-1、DART-Math、LIMO、Code-Feedback、OpenCodeInstruct。
- **中文 / ModelScope**：COIG-CQIA、MSAgent、DeepCtrl-SFT、Infinity Instruct、Alpaca-GPT4 中文、HC3-Chinese、Magpie-Qwen2-Pro-200K-Chinese、Chinese-DeepSeek-R1-Distill-110k-SFT、Seq-Monkey。

现已按 **SFT 数据集**、**RLHF 数据集**、**DPO 数据集** 三大类组织。


## 快速分类索引

| 分类 | 入口 | 亮点 |
| :----- | :----------- | :----------- |
| **通用 SFT** | [SFT 数据统计](#sft-数据统计) • [通用 SFT 详情](#通用-sft) | Alpaca、LIMA、UltraChat、OpenOrca、OpenHermes 2.5、WizardLM、Magpie、No Robots、Deita、WildChat、Capybara、Tulu 3、Smol-Smoltalk、PersonaHub |
| **代码 / 数学** | [SFT 数据统计](#sft-数据统计) • [代码 / 数学详情](#代码--数学) | Code Alpaca、MetaMathQA、OpenMathInstruct-1、DART-Math、LIMO、Code-Feedback、OpenCodeInstruct |
| **多语言 / 中文** | [SFT 数据统计](#sft-数据统计) • [多语言 / 中文详情](#多语言--中文) | Chinese-LLaMA-Alpaca、BELLE、COIG、COIG-CQIA、MSAgent、DeepCtrl-SFT、Infinity Instruct、Chinese-DeepSeek-R1-Distill、Seq-Monkey |
| **RLHF / 偏好** | [RLHF 数据统计](#rlhf-数据统计) • [RLHF 详情](#rlhf-数据集详情) | HH-RLHF、SHP、UltraFeedback、HelpSteer 系列 |
| **DPO** | [DPO 数据统计](#dpo-数据统计) • [DPO 详情](#dpo-数据集详情) | Orca-DPO-Pairs、UltraFeedback-Binarized、DPO-Mix-7K、Llama3-UltraFeedback-ArmoRM、Magpie DPO |
| **ModelScope 中文** | [ModelScope 中文指令微调数据集](#modelscope-中文指令微调数据集) | COIG-CQIA、MSAgent、DeepCtrl-SFT、Infinity Instruct、Chinese-DeepSeek-R1-Distill、Seq-Monkey |


## SFT 数据集

我们参考[这里](https://github.com/yaodongC/awesome-instruction-dataset) ([@yaodongC](https://github.com/yaodongC))，将收集到的数据集按照以下规则标注 Tags：

**(Lang) 语言标签**：
- EN: 英文指令数据集
- CN: 中文指令数据集
- ML: 多语言指令数据集

**(Task) 任务标签**：
- MT: 多任务数据集
- TS: 特定任务数据集

**(Gen) 生成方式**：
- HG: 人工生成数据集
- SI: Self-Instruct 生成数据集
- MIX: 混合数据集（人工 + 机器生成）
- COL: 数据集集合（由多个数据集组成）

### SFT 数据统计

| 数据集                                                                         | 数目      | Lang         | Task      | Gen        | 类型                                                                     | 来源                                      | 链接                                                                                       |
| :----------------------------------------------------------------------------- | :------- | :----------- | :-------- | :----------| :----------------------------------------------------------------------- | :---------------------------------------- | :---------------------------------------------------------------------------------------- |
| [Chain of Thought](https://github.com/google-research/FLAN)                    | 74771    | EN/CN        | MT        | HG         | CoT相关任务                                                               | 人在现有数据集上标注CoT                    | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chain-of-Thought)     |
| [GPT4all](https://github.com/nomic-ai/gpt4all)                                 | 806199   | EN           | MT        | COL        | 代码，故事，对话                                                           | GPT-3.5-turbo 蒸馏                       | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all)              |
| [GPTeacher](https://github.com/teknium1/GPTeacher)                             | 29013    | EN           | MT        | SI         | 通用，角色扮演，工具指令                                                   | GPT-4 & toolformer                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPTeacher)            |
| [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)       | 534610   | ML           | MT        | SI         | 多种nlp任务                                                               | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Guanaco)              |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)                      | 37175    | EN/CN        | TS        | MIX        | 对话评估                                                                  | gpt-3.5 或 人工                           | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/HC3)                  |
| [HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)      | 13k      | CN           | TS        | MIX        | 对话评估                                                                  | gpt-3.5 或 人工                           |                                                                                            |
| [alpaca](https://github.com/tatsu-lab/stanford_alpaca)                         | 52002    | EN           | MT        | SI         | 通用指令                                                                  | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca)               |
| [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)             | 52k      | EN           | MT        | SI         | 通用指令                                                                  | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca)               |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)          | 51k      | CN           | MT        | SI         | 通用指令                                                                  | text-davinci-003                         |                                                                                            |
| [Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM)             | 52k      | CN           | MT        | SI         | 通用指令                                                                  | text-davinci-003                         |                                                                                            |
| [Natural Instructions](https://github.com/allenai/natural-instructions)        | 5040134  | ML           | MT        | COL        | 多种nlp任务                                                               | 人工标注的数据集的收集                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Natural-Instructions) |
| [belle_cn](https://huggingface.co/BelleGroup)                                  | 1079517  | CN           | TS/MT     | SI         | 通用指令，数学推理，对话                                                   | text-davunci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/belle_cn)             |
| [instinwild](https://github.com/XueFuzhao/InstructionWild)                     | 52191    | EN/CN        | MT        | SI         | 生成，开放域问答，头脑风暴                                                 | text-davunci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instinwild)           |
| [华驼(HuaTuo)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)            | 8K       | CN           | TS        | SI         | 公开和自建的中文医学知识库                                                 | GPT3.5                                   |                                                                                            |
| [prosocial dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)   | 165681   | EN           | TS        | MIX        | 对话                                                                      | GPT-3改写问题，人工回复                    | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/prosocial-dialog)     |
| [finance_en](https://huggingface.co/datasets/gbharti/finance-alpaca)           | 68912    | EN           | TS        | COL        | 金融领域问答                                                              | GPT3.5                                   | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/)                     |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)                          | 78883588 | ML           | MT        | COL        | 多种nlp任务                                                               | 人工标注的数据集的收集                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/xP3)                  |
| [firefly](https://github.com/yangjianxin1/Firefly)                             | 1649398  | CN           | MT        | COL        | 23种nlp任务                                                               | 收集中文数据集，人工书写指令模板            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/firefly)              |
| [instruct](https://huggingface.co/datasets/swype/instruct)                     | 888969   | EN           | MT        | COL        | GPT4All，Alpaca和开源数据集的增强                                          | 使用AllenAI提供的nlp增强工具               | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instruct)             |
| [Code Alpaca](https://github.com/sahil280114/codealpaca)                       | 20022    | EN           | SI        | SI         | 代码生成，编辑，优化                                                       | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/CodeAlpaca)            |
| [Alpaca_GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)      | 52002    | EN/CN        | MT        | SI         | 通用指令                                                                  | GPT-4 生成的Alpaca数据                    | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpacaGPT4)            |
| [webGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)            | 18994    | EN           | TS        | MIX        | 信息检索问答                                                              | fine-tuned GPT-3 + 人工评估               | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/webGPT)                |
| [dolly 2.0](https://github.com/databrickslabs/dolly)                           | 15015    | EN           | TS        | HG         | 公开、封闭式问答、信息抽取、摘要生成、开放式构思、分类以及创意写作七类任务      | 人工标注                                  | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/dolly)                 |
| [mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry)                | 59.3K    | EN           | TS        | HG         | Dolly + HH-RLHF 组合                                                      | 人工标注                                  |                                                                                            |
| [baize](https://github.com/project-baize/baize-chatbot)                        | 653699   | EN           | MT        | COL        | Alpaca和多种问答任务                                                       | 人工标注的数据集的收集                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/baize)                 |
| [hh-rlhf](https://github.com/anthropics/hh-rlhf)                               | 284517   | EN           | TS        | MIX        | 对话                                                                      | RLHF models                              | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/hh-rlhf)               |
| [OIG(part)](https://laion.ai/blog/oig-dataset/)                                | 49237    | EN           | MT        | COL        | 多种nlp任务                                                               | 人工标注的数据集的收集和数据增强            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/OIG)                   |
| [GAOKAO](https://github.com/OpenLMLab/GAOKAO-Bench)                            | 2785     | CN           | MT        | COL        | 高考中的多选，填空等问题                                                   | 人工标注的数据集的收集                      | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GAOKAO)               |
| [camel](https://github.com/lightaime/camel)                                    | 760620   | EN           | MT        | SI         | 物理生物化学编程，数学，社会等领域的角色扮演对话                            | gpt-3.5-turbo 生成                        | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/camel)                 |
| [FLAN-Muffin](https://huggingface.co/datasets/Muennighoff/flan)                | 1764800  | EN           | MT        | COL        | 60种nlp任务                                                              | 人工标注的数据集的收集                      | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/FLAN-Muffin)           |
| [COIG](https://huggingface.co/datasets/BAAI/COIG)                              | 298428   | CN           | MT        | COL        | 考试，翻译，价值观指令数据集搜集，基于知识图谱的反事实对话                    | 自动化工具+人工验证                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/COIG)                 |
| [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)                          | 71446    | EN           | MT        | SI         | a collection of tool-related instructions                               | gpt-3.5-turbo                                | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/gpt4tools)             |
| [ShareChat](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)               | 1663241  | EN           | MT        | MIX        | general instruct                                                         | 收集ShareGPT                                 | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/ShareGPT)              |
| [Auto CoT](https://github.com/amazon-science/auto-cot)                         |          | EN           |           |            |                                                                          |                                            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Auto-CoT)              |
| [MOSS](https://github.com/OpenLMLab/MOSS)                                      | 1583595  | EN/CN        | SI        |            |                                                                          |                                            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/MOSS)                  |
| [ultrachat](https://github.com/thunlp/UltraChat)                               | 28247446 | EN           |           |            |                                                                          |                                            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/ultrachat)             |
| [StackLLaMA](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)    | todo     | EN           |           |            |                                                                          |                                            |                                                                                                 |
| [Self-Instruct](https://github.com/yizhongw/self-instruct)                     | 82 K     | EN           | SI        | SI         |                                                                          |                                            |                                                                                            |
| [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL)                | 100 w    |              | SI        | HG         | Zhihu data for training Open Assitant                                    |                                            |                                                                                            |
| [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)             | 385 k    | EN           | MT        | HG         |                                                                          | human preferences over responses           |                                                                                            |
| [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)          | 84.4k    | EN           | MT        | HG         | OpenAssistant Conversations Dataset (OASST1)                             | human-generated, human-annotated           |                                                                                            |
| [akoksal/LongForm](https://github.com/akoksal/LongForm)                        | 30k      | EN           | SI        | HG         |                                                                          | 从现有语料库选择文档并由 LLM 生成指令      |                                                                                            |
| [sail-sg/symbolic-instruction-tuning](https://github.com/sail-sg/symbolic-instruction-tuning) | 800K | ML   | SI        |            |                                                                          | Human Synthetic Examples                   |                                                                                            |
| 医疗问答 [michael-wzhu/PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE) | 110113   | CN           | SI        |            |                                                                          | 互联网医疗问诊问题，response 由 GPT-3.5 回答 |                                                                                            |
| [mbzuai-nlp/LaMini-LM](https://github.com/mbzuai-nlp/LaMini-LM)                | **2.58M**| EN           | MT        | SI         |                                                                          | 通过离线蒸馏从大型语言模型中提取知识       |                                                                                            |
| [pCLUE](https://github.com/CLUEbenchmark/pCLUE)                                | 120 万   |              |           |            |                                                                          |                                            |                                                                                            |
| [WizardLM](https://github.com/nlpxucan/WizardLM)                               | 70k-196k | EN           | MT        | SI         | 通用指令，通过 Evol-Instruct 提升复杂度                                     | GPT-4 / ChatGPT 演化生成                   | [下载](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k)            |
| [LIMA](https://huggingface.co/datasets/GAIR/lima)                              | 1k       | EN           | MT        | HG         | 高质量人工撰写指令                                                          | Stack Exchange、wikiHow、Reddit、人工作者   | [下载](https://huggingface.co/datasets/GAIR/lima)                                              |
| [UltraChat](https://github.com/thunlp/UltraChat)                               | 1.5M/200k| EN           | MT        | SI         | 多轮对话                                                                    | GPT-3.5-turbo 生成并过滤                    | [下载](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                           |
| [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)                 | 4.2M/550k| EN           | MT        | SI         | FLAN 风格推理轨迹                                                           | GPT-4 / GPT-3.5 增强 FLAN Collection       | [下载](https://huggingface.co/datasets/Open-Orca/SlimOrca)                                     |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)       | ~1M      | EN           | MT        | COL        | 精选多源混合：SlimOrca、Evol-Instruct、Platypus、ShareGPT 等               | 开源数据集精选                              | [下载](https://huggingface.co/datasets/teknium/OpenHermes-2.5)                                 |
| [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-3.2)           | 58.7k    | EN           | MT        | SI         | 通用、代码、角色扮演等                                                       | GPT-4 self-instruct                         | [下载](https://huggingface.co/datasets/jondurbin/airoboros-3.2)                                |
| [Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)         | 25k      | EN           | TS        | COL        | STEM 数学与科学                                                              | 精选数据集合并                              | [下载](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)                             |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)             | 395k     | EN           | TS        | SI         | 数学推理                                                                     | 从 GSM8K / MATH 自举生成                    | [下载](https://huggingface.co/datasets/meta-math/MetaMathQA)                                   |
| [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)           | 10k      | EN           | MT        | HG         | 人工精标跨 10 类指令                                                        | 人工标注                                    | [下载](https://huggingface.co/datasets/HuggingFaceH4/no_robots)                                |
| [Magpie](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-1M-v0.1)      | 1M/200k  | EN/CN        | MT        | SI         | 从对齐模型自动合成对齐数据                                                   | Llama-3 / Qwen2-Instruct 自动生成           | [下载](https://huggingface.co/datasets/Magpie-Align)                                           |
| [Infinity Instruct](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct) | 7M+ | EN/CN    | MT        | COL        | 大规模通用指令合集                                                           | 集合 + 合成                                 | [下载](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct)                         |
| [COIG-CQIA](https://modelscope.cn/datasets/m-a-p/COIG-CQIA)                    | 48k      | CN           | MT        | COL        | 高质量中文问答与文章                                                          | 中文互联网数据清洗 + 人工复核                | [下载](https://modelscope.cn/datasets/m-a-p/COIG-CQIA)                                         |
| [MSAgent / MSAgent-Bench](https://modelscope.cn/datasets/iic/MSAgent-Bench)    | 600k+    | CN/EN        | MT        | COL        | 工具调用与 Agent 能力                                                        | API 调用、多轮工具交互                       | [下载](https://modelscope.cn/datasets/iic/MSAgent-Bench)                                       |
| [DeepCtrl-SFT](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)      | 12M      | CN/EN        | MT        | MIX        | 安全的中英文通用 SFT 数据                                                    | 清洗 + 安全过滤                             | [下载](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)                              |
| [OpenCodeInstruct](https://huggingface.co/datasets/OpenCoder-llm/OpenCoder-llm)| 5M       | EN           | TS        | SI         | 大规模代码指令微调                                                           | 合成代码数据 + 执行验证                     | [下载](https://huggingface.co/datasets/OpenCoder-llm/OpenCoder-llm)                            |
| [Deita](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0)                | 10k      | EN           | MT        | COL        | 数据高效的高质量对齐 SFT 数据                                                | 复杂度+质量+多样性筛选                      | [下载](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0)                               |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M)                | 1M-4.8M  | ML           | MT        | HG         | 真实人类与 ChatGPT / 多模型对话                                              | Chatbot Arena 与 ChatGPT 真实交互           | [下载](https://huggingface.co/datasets/allenai/WildChat-1M)                                  |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara)                     | 15k      | EN           | MT        | SI         | 多轮推理对话                                                                 | Amplify-Instruct 扩展种子指令               | [下载](https://huggingface.co/datasets/LDJnr/Capybara)                                       |
| [Tulu 3 SFT Mix](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)   | ~939k    | EN           | MT        | COL        | Ai2 开放后训练 SFT 混合数据                                                  | 精选开源指令数据集                          | [下载](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)                           |
| [Smol-Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)       | 3.4M     | EN/ML        | MT        | COL        | SmolLM 系列指令数据                                                          | 多源合成 + 多语言                           | [下载](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)                              |
| [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub)          | 200k personas / 100k+ instruct | EN/CN | MT | SI | 基于 Persona 的合成指令数据                                                  | 10 亿 Persona 规模化生成                    | [下载](https://huggingface.co/datasets/proj-persona/PersonaHub)                              |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)| 1.8M     | EN           | TS        | SI         | 数学问题求解                                                                 | GSM8K / MATH 合成解答                       | [下载](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)                            |
| [DART-Math](https://huggingface.co/datasets/hkust-nlp/DART-Math)               | 600k+    | EN           | TS        | SI         | 难度感知数学指令微调                                                         | 难度控制拒绝采样                            | [下载](https://huggingface.co/datasets/hkust-nlp/DART-Math)                                  |
| [LIMO](https://huggingface.co/datasets/GAIR/LIMO)                              | 817      | EN           | TS        | COL        | 高质量推理激发数据                                                           | 精选 MATH/AIME/NuminaMath 专家 CoT          | [下载](https://huggingface.co/datasets/GAIR/LIMO)                                            |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)           | 200k+    | EN           | TS        | COL        | 代码指令 + 执行反馈                                                          | 多代码数据集合并 + 测试反馈                 | [下载](https://huggingface.co/datasets/m-a-p/Code-Feedback)                                  |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)       | 1.3T+ tokens | ML       | MT        | COL        | 高质量教育网页语料（预训练/持续预训练）                                       | FineWeb 教育内容过滤                        | [下载](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)                            |
| [Chinese-DeepSeek-R1-Distill-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k) | 110k | CN | MT | SI | 中文 DeepSeek-R1 蒸馏推理 SFT 数据                                           | 数学/考试/STEM/通用中文推理                 | [下载](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)        |
| [Seq-Monkey](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey)           | 100亿 tokens | CN       | MT        | COL        | 序列猴子中文预训练语料                                                       | 网页/百科/问答/博客/书籍/代码               | [下载](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey)                               |


### SFT 数据集详情

#### 通用 SFT

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

## [LIMA](https://arxiv.org/abs/2305.11206)
- [Dataset Link](https://huggingface.co/datasets/GAIR/lima)
- Summary: 1,000 carefully curated human-written instructions showing that "less is more for alignment". LIMA demonstrates that short fine-tuning on small, high-quality data can achieve competitive results with much larger synthetic datasets.
- Data generation model: `human generated`
- paper: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- Cost: N/A

## [UltraChat / UltraChat 200k](https://github.com/thunlp/UltraChat)
- [Dataset Link](https://huggingface.co/datasets/stingning/ultrachat)
- [Filtered Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- Summary: Large-scale multi-turn dialogue dataset generated by GPT-3.5-turbo. The HuggingFaceH4 release provides a sanitized, filtered 200k subset used in models like Zephyr.
- Data generation model: `GPT-3.5-turbo`
- paper: [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233)
- Cost: N/A

## [OpenOrca / SlimOrca](https://github.com/Open-Orca/OpenOrca)
- [Dataset Link](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [Cleaned Subset Link](https://huggingface.co/datasets/Open-Orca/SlimOrca)
- Summary: ~4.2M examples augmenting FLAN Collection prompts with GPT-4/GPT-3.5 completions and reasoning traces. SlimOrca is a cleaned, deduplicated subset focused on replicating Microsoft Orca-style reasoning.
- Data generation model: `GPT-4`, `GPT-3.5`
- paper: [OpenOrca: An Open Dataset of GPT Augmented FLAN Reasoning Traces](https://arxiv.org/abs/2308.07116)
- Cost: N/A

## [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [Dataset Link](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- Summary: A ~1M sample curated mixture of high-quality open-source instruction datasets including SlimOrca, Evol-Instruct, Platypus, ShareGPT, Airoboros, GPTeacher and others. Widely used for training generalist chat models.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

## [WizardLM / Evol-Instruct](https://github.com/nlpxucan/WizardLM)
- [Dataset Link 70k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k)
- [Dataset Link V2 196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k)
- Summary: Evol-Instruct iteratively rewrites simple instructions into more complex variants using LLMs. WizardLM applies this method to create complexity-evolved instruction data for training capable instruction followers.
- Data generation model: `GPT-4`, `ChatGPT`
- paper: [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
- Cost: N/A

## [Airoboros](https://huggingface.co/datasets/jondurbin/airoboros-3.2)
- [Dataset Link](https://huggingface.co/datasets/jondurbin/airoboros-3.2)
- Summary: Customizable self-instruct pipeline producing high-quality instruction data across general, code, roleplay and other domains. Version 3.2 contains ~58.7k samples.
- Data generation model: `GPT-4`
- paper: N/A
- Cost: N/A

## [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- Summary: 10k high-quality, fully human-curated instruction dataset modeled after OpenAI's InstructGPT paper. Covers 10 categories including generation, QA, coding, summarization and more.
- Data generation model: `human generated`
- paper: N/A
- Cost: N/A

## [Magpie](https://github.com/magpie-align/magpie)
- [Dataset Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-1M-v0.1)
- [Chinese Dataset Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese)
- Summary: Alignment data synthesis from scratch by prompting aligned LLMs with nothing—no seed data required. Magpie-Pro contains ~1M samples generated from Llama-3-70B-Instruct; a 200k Chinese variant is available from Qwen2.
- Data generation model: `Llama-3-Instruct`, `Qwen2-Instruct`
- paper: [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
- Cost: N/A

## [Deita](https://github.com/hkust-nlp/deita)
- [Dataset Link](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0)
- Summary: A 10K-sample data-efficient instruction tuning dataset selected by complexity, quality, and diversity scorers. Deita shows that principled data selection can match or exceed models trained on 10x more data.
- Data generation model: `COL`
- paper: [What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685)
- Cost: N/A

## [WildChat](https://wildchat.allen.ai/)
- [Dataset Link](https://huggingface.co/datasets/allenai/WildChat-1M)
- [Related Dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- Summary: Large-scale real-world human-ChatGPT / multi-model conversation logs (1M–4.8M). WildChat captures diverse, often longer and more complex user prompts than synthetic datasets; LMSYS-Chat-1M covers 25+ models and ~154 languages.
- Data generation model: `HG`
- paper: [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.14770) / [LMSYS-Chat-1M](https://arxiv.org/abs/2309.11998)
- Cost: N/A

## [Capybara](https://huggingface.co/datasets/LDJnr/Capybara)
- [Dataset Link](https://huggingface.co/datasets/LDJnr/Capybara)
- Summary: ~15K multi-turn reasoning conversations generated using Amplify-Instruct, expanding high-quality seed instructions into deep, multi-turn dialogues across science, math, logic, and culture.
- Data generation model: `GPT-4`
- paper: N/A
- Cost: N/A

## [Tulu 3 SFT Mix](https://github.com/allenai/open-instruct)
- [Dataset Link](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- Summary: ~939K open post-training SFT mixture curated by Ai2 for Tulu 3. Combines high-quality public instruction datasets into a single training blend.
- Data generation model: `COL`
- paper: [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.09601)
- Cost: N/A

## [Smol-Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Dataset Link 2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- Summary: Instruction dataset for the SmolLM family. smoltalk2 (~3.4M samples) adds OpenThoughts, Tulu 3, and multilingual data for training small but capable chat models.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

## [PersonaHub](https://github.com/tencent-ailab/persona-hub)
- [Dataset Link](https://huggingface.co/datasets/proj-persona/PersonaHub)
- Summary: 200K personas (toward 1B) released by Tencent AI Lab for persona-driven synthetic data generation. Configurations include instruction, math, reasoning, knowledge, NPC, and tool definitions.
- Data generation model: `SI`
- paper: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)
- Cost: N/A

#### 代码 / 数学

## [MetaMathQA](https://github.com/meta-math/MetaMath)
- [Dataset Link](https://huggingface.co/datasets/meta-math/MetaMathQA)
- Summary: ~395K math question-answer pairs bootstrapped from GSM8K and MATH via forward/backward reasoning and rephrasing. Used to significantly improve mathematical reasoning in LLMs.
- Data generation model: `GPT-3.5`
- paper: [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://arxiv.org/abs/2309.12284)
- Cost: N/A

## [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)
- [Dataset Link](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)
- Summary: 1.8M math instruction tuning pairs with synthetic solutions for GSM8K and MATH, designed to improve open math reasoning.
- Data generation model: `GPT-4`
- paper: [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176)
- Cost: N/A

## [DART-Math](https://github.com/hkust-nlp/dart-math)
- [Dataset Link](https://huggingface.co/datasets/hkust-nlp/DART-Math)
- Summary: Difficulty-Aware Rejection Tuning dataset for math. Generates and filters reasoning trajectories by difficulty, yielding 600K+ high-quality math instruction examples.
- Data generation model: `GPT-4`
- paper: [DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving](https://arxiv.org/abs/2407.13690)
- Cost: N/A

## [LIMO](https://github.com/GAIR-NLP/LIMO)
- [Dataset Link](https://huggingface.co/datasets/GAIR/LIMO)
- Summary: Only 817 curated high-quality reasoning examples that elicit strong complex reasoning (AIME 57%, MATH 95%) from a 32B base model, demonstrating "less is more for reasoning".
- Data generation model: `COL`
- paper: [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)
- Cost: N/A

## [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)
- [Dataset Link](https://huggingface.co/datasets/m-a-p/Code-Feedback)
- [Filtered Version](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)
- Summary: 200K+ code instruction pairs with execution feedback from the M-A-P project. Used to train code models with test-driven verification signals.
- Data generation model: `COL`
- paper: [M-A-P: Introducing High-Quality Code-Feedback Dataset](https://arxiv.org/abs/2407.12883)
- Cost: N/A

## [OpenCodeInstruct](https://github.com/OpenCoder-llm/OpenCoder-llm)
- [Dataset Link](https://huggingface.co/datasets/OpenCoder-llm/OpenCoder-llm)
- Summary: 5 million code instruction tuning samples with execution-based verification, one of the largest open code instruction datasets.
- Data generation model: `COL`
- paper: [OpenCoder: The Open Cookbook for Top-Tier Code LLMs](https://arxiv.org/abs/2411.04905)
- Cost: N/A

#### 多语言 / 中文

## [xP3](https://huggingface.co/datasets/bigscience/xP3)
- [Dataset Link](https://huggingface.co/datasets/bigscience/xP3)
- Summary: [Prompt-resource] xP3 (Crosslingual Public Pool of Prompts) is a collection of prompts & datasets across 46 of languages & 16 NLP tasks.
- Data generation model: N/A
- paper: [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786)
- Cost: N/A

## [firefly](https://github.com/yangjianxin1/Firefly)
- [Dataset Link](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- Summary: Chinese datasets of 23 tasks combined with human-written instruction templates.
- Data generation model: N/A
- paper: N/A
- Cost: N/A

## [COIG](https://huggingface.co/datasets/BAAI/COIG)
- [Dataset Link](https://huggingface.co/datasets/BAAI/COIG)
- Summary: Chinese Open Instruction Generalist dataset containing translated instructions, exam instructions, value alignment instructions, counterfactual correction multi-round chat, and Leetcode instructions.
- Data generation model: `COL`
- paper: [Chinese Open Instruction Generalist: A Preliminary Release](https://arxiv.org/abs/2304.07987)
- Cost: N/A

## [Infinity Instruct](https://github.com/FlagOpen/Infinity-Instruct)
- [Dataset Link (ModelScope)](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct)
- [Dataset Link (HuggingFace)](https://huggingface.co/datasets/AI-ModelScope/Infinity-Instruct)
- Summary: Large-scale general instruction dataset (7M+) released by BAAI / FlagOpen for both Chinese and English instruction tuning.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A

## [COIG-CQIA](https://github.com/CSJianYang/COIG-CQIA)
- [Dataset Link](https://huggingface.co/datasets/m-a-p/COIG-CQIA)
- [ModelScope Link](https://modelscope.cn/datasets/m-a-p/COIG-CQIA)
- Summary: High-quality Chinese instruction fine-tuning dataset following the "quality is all you need" philosophy. Built from Chinese internet Q&A and articles with deep cleansing, restructuring, and manual review.
- Data generation model: `COL`
- paper: [COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning](https://arxiv.org/abs/2403.18058)
- Cost: N/A

## [MSAgent / MSAgent-Bench](https://github.com/modelscope/modelscope-agent)
- [Dataset Link](https://huggingface.co/datasets/iic/MSAgent-Bench)
- [ModelScope Link](https://modelscope.cn/datasets/iic/MSAgent-Bench)
- Summary: Chinese/English agent tool-use dataset for training LLMs to call APIs and use external tools in multi-turn conversations.
- Data generation model: `MIX`
- paper: N/A
- Cost: N/A

## [DeepCtrl-SFT](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [ModelScope Link](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- Summary: 12M safe Chinese/English general-domain SFT entries (~3B tokens). Designed for small-scale Chinese LLM instruction fine-tuning with safety filtering.
- Data generation model: `MIX`
- paper: N/A
- Cost: N/A

## [Chinese-DeepSeek-R1-Distill-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- [Dataset Link](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k)
- [ModelScope Link](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k)
- Summary: 110K Chinese reasoning SFT samples distilled from the full DeepSeek-R1-671B model. Covers math (~36.5K), exam (~2.4K), STEM (~12K), and general Chinese reasoning (~58K).
- Data generation model: `DeepSeek-R1-671B`
- paper: N/A
- Cost: N/A

## [Seq-Monkey](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey)
- [ModelScope Link](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey)
- Summary: ~10B token Chinese pre-training corpus released by Mobvoi, including web, encyclopedia, Q&A, blogs, books, and code. Mainly for pre-training/continued pre-training; often combined with instruction data for full Chinese LLM training.
- Data generation model: `COL`
- paper: N/A
- Cost: N/A


## RLHF / 偏好数据集

### RLHF 数据统计

| 数据集 | 规模 | 语言 | 说明 | 链接 |
| :----- | :--- | :--- | :--- | :--- |
| webgpt_comparisons | 19,578 | EN | WebGPT 论文中用于奖励模型训练的人类偏好比较 | [下载](https://huggingface.co/datasets/openai/webgpt_comparisons) |
| Stanford SHP | 385k | EN | Reddit 上 18 个领域的人类偏好 | [下载](https://huggingface.co/datasets/stanfordnlp/SHP) |
| rlhf-reward-datasets | 76.3k | EN | RLHF 奖励模型数据集 | [下载](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets) |
| Dahoas/full-hh-rlhf | 112k | EN | Anthropic HH 数据集重排为 prompt/chosen/rejected | [下载](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) |
| Dahoas/synthetic-instruct-gptj-pairwise | - | EN | GPT-J 合成指令偏好对 | [下载](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) |
| Dahoas/rm-static | 76.3k | EN | hh-static 拆分用于奖励模型训练 | [下载](https://huggingface.co/datasets/Dahoas/rm-static) |
| Anthropic HH-RLHF | 22k-284k | EN | 有用性与无害性偏好人机对话 | [下载](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| GPT-4-LLM comparison | 52k | EN | GPT-4 评估的 Alpaca 三模型排序 | [项目](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) |
| thu-coai/Safety-Prompts | 100k | CN | 中文安全 prompts，用于对齐人类价值观 | [下载](https://huggingface.co/datasets/thu-coai/Safety-Prompts) |
| HC3 Comparison | 24.3K | EN/CN | 人类与 ChatGPT 回答对比 | [下载](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |
| UltraFeedback | 64k | EN | 多维度反馈（指令遵循、真实性、有用性等） | [下载](https://huggingface.co/datasets/openbmb/UltraFeedback) |
| HelpSteer | 37k | EN | 多属性 helpfulness 评分 | [下载](https://huggingface.co/datasets/nvidia/HelpSteer) |
| HelpSteer2 | 10k | EN | 奖励模型训练偏好数据 | [下载](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| HelpSteer2-Preference | 10k | EN | 带人工解释的偏好对 | [下载](https://huggingface.co/datasets/nvidia/HelpSteer2-Preference) |
| HelpSteer3-Preference | 40k+ | EN | 跨 STEM、代码、多语言的开放式人类偏好数据 | [下载](https://huggingface.co/datasets/nvidia/HelpSteer3-Preference) |

### RLHF 数据集详情

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

## [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [Paper/Project Link](https://arxiv.org/abs/2310.01377)
- [Dataset Link](https://huggingface.co/datasets/openbmb/UltraFeedback)

A large-scale, multi-aspect feedback dataset for training reward models and DPO. It contains ~64k prompts with multiple ranked responses annotated for instruction-following, truthfulness, honesty and helpfulness.

## [HelpSteer / HelpSteer2 / HelpSteer2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [Paper/Project Link](https://arxiv.org/abs/2311.09528)
- [Dataset Link](https://huggingface.co/datasets/nvidia/HelpSteer)
- [Dataset Link 2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [Dataset Link 2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2-Preference)

NVIDIA's family of multi-attribute helpfulness datasets. HelpSteer provides fine-grained ratings across helpfulness, correctness, coherence, complexity and verbosity. HelpSteer2 adds preference pairs for reward model training; HelpSteer2-Preference includes human-written justifications comparing Bradley-Terry and regression reward modeling. HelpSteer3-Preference expands to 40k+ diverse real-world applications including STEM, coding, and multilingual tasks.


## DPO 偏好优化数据集

### DPO 数据统计

| 数据集 | 规模 | 语言 | 说明 | 链接 |
| :----- | :--- | :--- | :--- | :--- |
| Magpie-Pro-DPO | 100k | EN | 对齐模型合成的 DPO 偏好对 | [下载](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-DPO-100K-v0.1) |
| Orca-DPO-Pairs | 12k / 6k | EN | GPT-4 Orca 推理轨迹的 DPO 偏好对；Argilla 清洗版约 6k | [下载](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs) |
| UltraFeedback-Binarized | 61k-230k | EN | UltraFeedback 二值化的 chosen/rejected 偏好对 | [下载](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |
| DPO-Mix-7K | 7.5k | EN | Argilla 精选的 Capybara + Orca + UltraFeedback 高质量 DPO 混合 | [下载](https://huggingface.co/datasets/argilla/dpo-mix-7k) |
| Llama3-UltraFeedback-ArmoRM | 60k | EN | 每条 prompt 5 个 Llama-3-SFT 回答，ArmoRM 排序 | [下载](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) |
| ChatML-DPO-Pairs | 12k | EN | ChatML 格式 Orca DPO 对（ChatGPT chosen / Llama-2 rejected） | [下载](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs) |
| Magpie-Air-DPO | 100k | EN | Llama-3-8B-Instruct 合成的 DPO 偏好对 | [下载](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1) |
| Tulu 3 Preference | 20k-200k | EN | Ai2 Tulu 3 后训练偏好数据 | [下载](https://huggingface.co/datasets/allenai/tulu-3-pref-mixture) |

### DPO 数据集详情

## [Orca-DPO-Pairs](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs)
- [Alternative Source](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [Cleaned Version](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs)

~12k DPO preference pairs derived from GPT-4 augmented Orca/FLAN reasoning traces. Intel released a widely used variant; Argilla later cleaned it to ~6k high-quality pairs, swapping/rejecting mislabeled examples.

## [UltraFeedback-Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [Dataset Link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

61k–230k preference pairs obtained by binarizing UltraFeedback scores into chosen/rejected responses. A standard DPO dataset used in Zephyr, Tulu 3, and many open alignment recipes.

## [DPO-Mix-7K](https://huggingface.co/datasets/argilla/dpo-mix-7k)
- [Dataset Link](https://huggingface.co/datasets/argilla/dpo-mix-7k)

A small, curated cocktail (~7.5k pairs) mixing high-scoring chosen responses from Capybara, Intel Orca, and UltraFeedback. Designed for efficient, high-quality DPO training.

## [Llama3-UltraFeedback-ArmoRM](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm)
- [Dataset Link](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm)

~60k prompts each with five Llama-3-SFT responses ranked by the ArmoRM reward model. Supports advanced preference optimization research beyond binary pairs.

## [ChatML-DPO-Pairs](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs)
- [Dataset Link](https://huggingface.co/datasets/mlabonne/chatml-dpo-pairs)

~12k ChatML-formatted preference pairs based on Intel/orca_dpo_pairs, with ChatGPT responses as chosen and Llama-2-13b-chat responses as rejected.

## [Magpie-Air-DPO](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1)
- [Dataset Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-DPO-100K-v0.1)

~100k synthetic DPO preference pairs generated from Llama-3-8B-Instruct using the Magpie-Air pipeline.

## [Tulu 3 Preference](https://github.com/allenai/open-instruct)
- [Dataset Link](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)
- [Preference Mixture](https://huggingface.co/datasets/allenai/tulu-3-pref-mixture)

Ai2's Tulu 3 preference data (~20k–200k) including persona-based instruction-following preferences and a full preference mixture for open post-training.


## ModelScope 中文指令微调数据集

魔搭社区 (ModelScope) 是国内最活跃的大模型开源平台之一，汇集了丰富的中文指令微调与 Agent 数据集。推荐重点关注：

| 数据集 | 链接 | 规模 | 说明 |
| :----- | :--- | :--- | :--- |
| **COIG-CQIA** | [modelscope/m-a-p/COIG-CQIA](https://modelscope.cn/datasets/m-a-p/COIG-CQIA) | 48k | 高质量中文指令，遵循“Quality is All You Need”理念 |
| **MSAgent / MSAgent-Bench** | [modelscope/iic/MSAgent-Bench](https://modelscope.cn/datasets/iic/MSAgent-Bench) | 600k+ | 中英文工具调用与 Agent 训练数据 |
| **DeepCtrl-SFT** | [modelscope/deepctrl/deepctrl-sft-data](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data) | 1200万 | 安全清洗的中英文通用 SFT 数据，约 3B tokens |
| **Infinity Instruct** | [modelscope/AI-ModelScope/Infinity-Instruct](https://modelscope.cn/datasets/AI-ModelScope/Infinity-Instruct) | 700万+ | BAAI/FlagOpen 发布的大规模中英文指令集合 |
| **Alpaca-GPT4 中文** | [modelscope/AI-ModelScope/alpaca-gpt4-data-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh) | 52k | GPT-4 生成的中文 Alpaca 指令数据 |
| **HC3-Chinese** | [modelscope/simpleai/HC3-Chinese](https://modelscope.cn/datasets/simpleai/HC3-Chinese) | 39,781 | 中文人类与 ChatGPT 回答对比语料 |
| **OpenHermes 2.5** | [modelscope/swift/OpenHermes-2.5](https://modelscope.cn/datasets/swift/OpenHermes-2.5) | ~1M | 高质量通用指令混合数据 |
| **Magpie-Qwen2-Pro-200K-Chinese** | [HuggingFace/ModelScope](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese) | 200k | 基于 Qwen2 的中文对齐合成数据 |
| **Chinese-DeepSeek-R1-Distill-110k-SFT** | [modelscope/liucong/Chinese-DeepSeek-R1-Distill-data-110k](https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k) | 110k | 中文 DeepSeek-R1-671B 蒸馏推理 SFT 数据 |
| **Seq-Monkey** | [modelscope/ddzhu123/seq-monkey](https://modelscope.cn/datasets/ddzhu123/seq-monkey) | 100亿 tokens | 出门问问开源的中文通用预训练语料 |

> 提示：可通过 [ms-swift](https://github.com/modelscope/ms-swift) 框架一键使用上述数据集进行微调。


## 无许可信息的数据集

## [alespalla/chatbot_instruction_prompts](https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts)

- Summary: A compilation of `tatsu-lab/alpaca` ,`Dahoas/instruct-human-assistant-prompt` ,`allenai/prosocial-dialog`
- Data generation model: N/A
- paper: N/A
- Cost: N/A


## 贡献指南

Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](#贡献模板) for instructions in contribution.

### 贡献模板

Append the new project at the end of file:

```markdown
[{Project-name}/{Dataset-name}](https://github.com/link/to/project)

- [paper/project link](link)
- [dataset link](link)
- Related work: (if applicable)

Some introductions ...
```


## 许可证

`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.


## 参考
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
