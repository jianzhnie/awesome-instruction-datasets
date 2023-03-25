## Summary of awesome Prompt datasets

|                Datasets/Projects                | Organization/Author | Introduction                                                 |
| :---------------------------------------------: | :-----------------: | :----------------------------------------------------------- |
| Natural Instruction / Super-Natural Instruction |      Allen AI       | 包含61个NLP任务（Natural Instruction）和1600个NLP任务（Super-Natural Instruction）的指令数据 |
|                PromptSource / P3                |     BigScience      | 包含270个NLP任务的2000多个prompt模版（PromptSource）和规模在100M-1B之间的P3数据集 |
|                      xMTF                       |     BigScience      | 包含13个NLP任务、46种语言的多语言prompt数据                  |
|                     HH-RLHF                     |      Anthropic      | 旨在训练Helpful and Harmless（HH）的LLMs的RLHF数据集         |
|              Unnatural Instruction              |     orhonovich      | 使用GPT3生成64k的instruction prompt数据，经改写后得到240k条instruction数据 |
|                  Self-Instruct                  |      yizhongw       | 使用LLMs生成prompt进行instruct-tuning的方法，引入Task pool和Quality filtering等概念 |
|                   UnifiedSKG                    |         HKU         | 在Text-to-Text框架中加入knowledge grounding，将结构化数据序列化并嵌入到prompt中 |
|                 Flan Collection                 |       Google        | 将Flan 2021数据与一些开源的instruction数据（P3，super-natural instruction等）进行合并 |
|                  InstructDial                   |    prakharguptaz    | 在特定的一种任务类型（对话指令）上进行指令微调的尝试         |
|                     Alpaca                      |      Stanford       | 53k data, very powerful performance (GPT-3.5 level).         |

## List of awesome Prompt datasets

### Natural Instruction / Super-Natural Instruction

- [Paper/Project link](https://aclanthology.org/2022.acl-long.244.pdf)
- [Dataset link](https://instructions.apps.allenai.org/)

Allen AI 是第一批尝试Instruction做prompt并微调LLMs的机构。在Natural Instruction论文里可以基本了解instruction的标注思路.

在其提出的数据集中，包含了61和不同的NLP tasks。

Super-Natural Instruction 是Natural Instruction的超级加量版，其包含了超过1600个不同的NLP任务，光是不同**种类**的NLP任务（例如：分类，抽取，序列标注）就超过76个。

### PromptSource / P3

- [Paper/Project Link](https://github.com/bigscience-workshop/promptsource)

- [Dataset Link](https://huggingface.co/datasets/bigscience/P3)

BigScience由Hugging Face和法国CNRS，IDRIS，GENCI等联合组织，是当下最大的开源LLMs组织之一。

BigScience在2021年末开发了PromptSource项目，开源了一系列工具toolkits，帮助研究者基于现有NLP任务构建prompt。截止目前，PromptSource项目包含了270个NLP任务的超过2000个prompt模版。

在此基础上，BigScience构建了P3数据集。在Hugging Face Hub上你可以找到P3数据，P3的数据规模在100M-1B之间。

### xMTF - BigScience

- [Project Link](https://arxiv.org/pdf/2211.01786.pdf)

- [Dataset Link](https://github.com/bigscience-workshop/xmtf)

BigScience在英语prompt的基础上，扩展其prompt到多种非英语语言。

该项目包含了13个NLP任务，并采用了46个不同的语言的版本。对应的prompt包含的语种个数不定。

在multilingual的基础上微调后，BLOOM和T0都变现出了理想的多语言能力。

### HH-RLHF - Anthropic

- [Paper/Project Link](https://arxiv.org/pdf/2204.05862.pdf)

- [Dataset Link](https://huggingface.co/datasets/Anthropic/hh-rlhf)

Anthropic公司旗下的Claud是ChatGPT的主要竞品之一。

Anthropic开源了其在自己产品线中使用的RLHF数据集。

HH-RLHF项目的初衷在于训练Helpful and Harmless（HH）的LLMs。故该项目除了回复质量外，是否为有害信息也体现在了其human feedback中。

论文中记录了如何使用RLHF数据Align模型的behaviour到人类的价值观上，同时记录了数据集的构建方式和标准。

### Unnatural Instruction

- [Paper/Project Link](https://arxiv.org/pdf/2212.09689.pdf)

- [Dataset Link](https://github.com/orhonovich/unnatural-instructions)

使用LLMs自主生成instruction数据是instruct-tuning领域较为活跃的一个方向。

Unnatural Instruction使用GPT3 (text-davinci-002)生成了64k的instruction prompt数据。并使用同样的模型将64k的prompt进行改写，最终得到了240k条instruction数据。

论文中显示，在Instruct-Tuning中LLMs自主生成的prompt表现出了良好的效果，甚至超过了在P3等数据上进行微调的T0等模型。

### Self-Instruct

- [Paper/Project Link](https://arxiv.org/pdf/2212.10560.pdf)

- [Dataset Link](https://github.com/yizhongw/self-instruct)

Self-Instruct同样是使用LLMs生成prompt进行instruct-tuning的思路。不过使用了更fine-grained的生成流程。

Task pool和Quality filtering等概念被引入，部分缓解了self-intrauct类型数据的noise问题。

### UnifiedSKG - HKU

[Paper/Project Link](https://arxiv.org/pdf/2201.05966.pdf)

[DataSet Link](https://unifiedskg.com/)

UnifiedSKG在Text-to-Text的框架中加入了knowledge grounding，也就是在prompt-output的框架中，加入了结构化数据做辅助。

举个例子，某些NLP任务非常依赖结构化的知识库/数据库。UnifiedSKG的思路是将需要的数据库序列化，并嵌入到prompt中。如下图所示。

UnifiedSKG代表了LLMs领域中尝试使用结构化知识增强性能的一个方向。

### Flan Collection - Google

- [Paper/Project Link](https://arxiv.org/pdf/2301.13688.pdf)
- [Dataset Link](https://github.com/google-research/FLAN/tree/main/flan/v2)

Google在这个项目中将自己的Flan 2021数据与一些开源的instruction数据（P3，super-natural instruction 等）进行了合并。

在Flan Collection的论文中，google也总结了Flan系列模型训练/推理中的一些关键点，可能会有不错的参考价值。

### InstructDial

- [Paper/Project Link](https://arxiv.org/pdf/2205.12673.pdf)

- [Dataset Link](https://github.com/prakharguptaz/Instructdial/tree/main/datasets)

InstructDial是在特定的一种任务类型上进行指令微调的尝试。实验结果表明，在对话指令数据上微调后，模型在对话任务上的表现强于在超大规模任务集上的结果。

### Alpaca -Stanford

- [Paper/Project Link](https://github.com/tatsu-lab/stanford_alpaca)

- [Dataset Link](https://github.com/tatsu-lab/stanford_alpaca)

Stanford release的Alpaca是在Meta Ai LLaMA模型基础上进行instruct-tuning的微调模型。

Alpaca使用GPT-3.5自动生成了52k的指令数据，并用其微调LLaMA模型。实验结果表明，其能够达到/甚至超过GPT-3.5在一些任务上的效果。

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License
`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.