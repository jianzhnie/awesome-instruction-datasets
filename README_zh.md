
<div align="center">
# Awesome Prompt datasets  for Chinese[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

</div>

<div align="center">
[English](README.md) | 中文
</div>


这是一个用于中文指令调整的 AWESOME 数据集合集。

通过使用指令进行微调，以提高 LLM（大型语言模型）的性能成为了一个趋势。随着以数据为中心的 AI 越来越受欢迎，我们需要更高质量的数据集来训练我们的模型。

在这里，你可以找到一些开源的中文指令数据的AWESOME 列表。


## 数据集合 (Data Collection)  


收集数据集的相对大小如下图所示:

我们参考[这里](https://github.com/yaodongC/awesome-instruction-dataset) ([@yaodongC](https://github.com/yaodongC)), 将收集到的数据集按照以下规则标注Tags：

(Lang)Lingual-Tags:
- EN: Instruction datasets in English
- CN: Instruction datasets in Chinese
- ML: [Multi-lingual] Instruction datasets in multiple languages

(Task)Task-Tags:
- MT: [Multi-task] Datasets containing multiple tasks
- TS: [Task-specific] Datasets tailored for specific tasks

(Gen)Generation-method:
- HG: [Human Generated Dataset] Datasets created by humans
- SI: [Self-Instruct] Datasets generated using self-instruct methods
- MIX: [Mixed Dataset] Dataset contains both human and machine generated data
- COL: [Collection of Dataset] Dataset made from a collection of other datasets

### 数据统计
| 数据集                                                                         | 数目      | Lang         | Task      | Gen        | 类型                                                                     | 来源                                      | 链接                                                                                       |
| :----------------------------------------------------------------------------- | :------- | :----------- | :-------- | :----------| :----------------------------------------------------------------------- | :---------------------------------------- | :---------------------------------------------------------------------------------------- |
| [Chain of Thought](https://github.com/google-research/FLAN)                    | 74771    | EN/CN        | MT        | HG         | CoT相关任务                                                               | 人在现有数据集上标注CoT                    | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chain-of-Thought)     |
| [GPT4all](https://github.com/nomic-ai/gpt4all)                                 | 806199   | EN           | MT        | COL        | 代码，故事，对话                                                           | GPT-3.5-turbo 蒸馏                       | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all)              |
| [GPTeacher](https://github.com/teknium1/GPTeacher)                             | 29013    | EN           | MT        | SI         | 通用，角色扮演，工具指令                                                   | GPT-4 & toolformer                        | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPTeacher)            |
| [Guanaco](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)       | 534610   | ML           | MT        | SI         | 多种nlp任务                                                               | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Guanaco)              |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)                      | 37175    | EN/CN        | TS        | MIX        | 对话评估                                                                  | gpt-3.5 或 人工                           | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/HC3)                  |
| [alpaca](https://github.com/tatsu-lab/stanford_alpaca)                         | 52002    | EN           | MT        | SI         | 通用指令                                                                  | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca)               |
| [Natural Instructions](https://github.com/allenai/natural-instructions)        | 5040134  | ML           | MT        | COL        | 多种nlp任务                                                               | 人工标注的数据集的收集                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Natural-Instructions) |
| [belle_cn](https://huggingface.co/BelleGroup)                                  | 1079517  | CN           | TS/MT     | SI         | 通用指令，数学推理，对话                                                   | text-davunci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/belle_cn)             |
| [instinwild](https://github.com/XueFuzhao/InstructionWild)                     | 52191    | EN/CN        | MT        | SI         | 生成，开放域问答，头脑风暴                                                 | text-davunci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instinwild)           |
| [prosocial dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)   | 165681   | EN           | TS        | MIX        | 对话                                                                     | GPT-3改写问题，人工回复                    | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/prosocial-dialog)     |
| [finance_en](https://huggingface.co/datasets/gbharti/finance-alpaca)           | 68912    | EN           | TS        | COL        | 金融领域问答                                                              | GPT3.5                                   | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/)                     |
| [xP3](https://huggingface.co/datasets/bigscience/xP3)                          | 78883588 | ML           | MT        | COL        | 多种nlp任务                                                               | 人工标注的数据集的收集                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/xP3)                  |
| [firefly](https://github.com/yangjianxin1/Firefly)                             | 1649398  | CN           | MT        | COL        | 23种nlp任务                                                               | 收集中文数据集，人工书写指令模板            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/firefly)              |
| [instruct](https://huggingface.co/datasets/swype/instruct)                     | 888969   | EN           | MT        | COL        | GPT4All，Alpaca和开源数据集的增强                                          | 使用AllenAI提供的nlp增强工具               | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/instruct)             |
| [Code Alpaca](https://github.com/sahil280114/codealpaca)                       | 20022    | EN           | SI        | SI         | 代码生成，编辑，优化                                                       | text-davinci-003                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/CodeAlpaca)            |
| [Alpaca_GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)      | 52002    | EN/CN        | MT        | SI         | 通用指令                                                                  | GPT-4 生成的Alpaca数据                    | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpacaGPT4)            |
| [webGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)            | 18994    | EN           | TS        | MIX        | 信息检索问答                                                              | fine-tuned GPT-3 + 人工评估               | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/webGPT)                |
| [dolly 2.0](https://github.com/databrickslabs/dolly)                           | 15015    | EN           | TS        | HG         | 公开、封闭式问答、信息抽取、摘要生成、开放式构思、分类以及创意写作七类任务      | 人工标注                                  | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/dolly)                 |
| [baize](https://github.com/project-baize/baize-chatbot)                        | 653699   | EN           | MT        | COL        | Alpaca和多种问答任务                                                       | 人工标注的数据集的收集                     | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/baize)                 |
| [hh-rlhf](https://github.com/anthropics/hh-rlhf)                               | 284517   | EN           | TS        | MIX        | 对话                                                                      | RLHF models                              | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/hh-rlhf)               |
| [OIG(part)](https://laion.ai/blog/oig-dataset/)                                | 49237    | EN           | MT        | COL        | 多种nlp任务                                                               | 人工标注的数据集的收集和数据增强            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/OIG)                   |
| [GAOKAO](https://github.com/OpenLMLab/GAOKAO-Bench)                            | 2785     | CN           | MT        | COL        | 高考中的多选，填空等问题                                                   | 人工标注的数据集的收集                      | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GAOKAO)               |
| [camel](https://github.com/lightaime/camel)                                    | 760620   | EN           | MT        | SI         | 物理生物化学编程，数学，社会等领域的角色扮演对话人工标注的数据集的收集         | gpt-3.5-turbo 生成                        | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/camel)                 |
| [FLAN-Muffin](https://huggingface.co/datasets/Muennighoff/flan)                | 1764800  | EN           | MT        | COL        | 60种nlp任务                                                              | 人工标注的数据集的收集                      | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/FLAN-Muffin)           |
| [COIG](https://huggingface.co/datasets/BAAI/COIG)                              | 298428   | CN           | MT        | COL        | 考试，翻译，价值观指令数据集搜集，基于知识图谱的反事实对话                    | 自动化工具+人工验证                         | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/COIG)                 |
| [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)                          | 71446    | EN           | MT        | SI         | a collection of tool-related instructions                               | gpt-3.5-turbo                                | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/gpt4tools)             |
| [ShareChat](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)               | 1663241  | EN           | MT        | MIX        | general instruct                                                         | 收集ShareGPT                                 | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/ShareGPT)              |
| [Auto CoT](https://github.com/amazon-science/auto-cot)                         |          | EN           |           |            |                                                                          |                                            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Auto-CoT)              |
| [MOSS](https://github.com/OpenLMLab/MOSS)                                      | 1583595  | EN/CN        | SI        |            |                                                                          |                                            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/MOSS)                  |
| [ultrachat](https://github.com/thunlp/UltraChat)                               | 28247446 | EN           |           |            |                                                                          |                                            | [下载](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/ultrachat)             |
| [StackLLaMA](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)    | todo     | EN           |           |            |                                                                          |                                            |                                                                                                 |



## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License
`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.