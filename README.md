
<div align="center">

# Awesome Prompt datasets

</div>

<div align="center">

[中文](README_zh.md) | English
</div>


## [Summary](https://jianzhnie.github.io/machine-learning-wiki/#/ai-general/chatgpt/awe_prompt?id=summary-of-awesome-prompt-datasets)

| Datasets/Projects                            | Organization/Author | Introduction                                                                                                           |
| :---------------------------------------------- | :------------------ | :--------------------------------------------------------------------------------------------------------------------- |
| Natural Instruction / Super-Natural Instruction | Allen AI            | Contains instruction data of 61 NLP tasks (Natural Instruction) and 1600 NLP tasks (Super-Natural Instruction)         |
| PromptSource / P3                               | BigScience          | More than 2,000 prompt templates (PromptSource) containing 270 NLP tasks and a P3 dataset with a scale between 100M-1B |
| xMTF                                            | BigScience          | Contains 13 NLP tasks and multilingual prompt data in 46 languages                                                     |
| HH-RLHF                                         | Anthropic           | RLHF dataset designed to train Helpful and Harmless (HH) LLMs                                                          |
| Unnatural Instruction                           | orhonovich          | Use GPT3 to generate 64k instruction prompt data, and get 240k instruction data after rewriting                        |
| Self-Instruct                                   | yizhongw            | Using LLMs to generate prompts for instruction-tuning, introducing concepts such as Task pool and Quality filtering    |
| UnifiedSKG                                      | HKU                 | Add knowledge grounding to Text-to-Text framework, serialize and embed structured data into prompt                     |
| Flan Collection                                 | Google              | Merge Flan 2021 data with some open source instruction data (P3, super-natural instruction, etc.)                      |
| InstructDial                                    | prakharguptaz       | Attempts to fine-tune instructions on a specific task type (dialogue instructions)                                     |
| Alpaca                                          | Stanford            | 53k data, very powerful performance (GPT-3.5 level).                                                                   |

## The template

Append the new project at the end of file
```shell

[{Project-name}/{Dataset-name}]{https://github.com/link/to/project}

- [paper/project link](link)
- [dataset link](link)

Some introductions ...
```


## The List

### Natural Instruction / Super-Natural Instruction

- [Paper/Project link](https://aclanthology.org/2022.acl-long.244.pdf)
- [Dataset link](https://instructions.apps.allenai.org/)

Allen AI is the first organization to try Instruction as a prompt and fine-tune LLMs. In the Natural Instruction paper, you can basically understand the labeling ideas of the instruction.

In its proposed dataset, 61 and different NLP tasks are included.

Super-Natural Instruction is a super-intensive version of Natural Instruction, which contains more than 1,600 different NLP tasks, and there are more than 76 different **types of NLP tasks (such as: classification, extraction, sequence labeling).**

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

[Paper/Project Link](https://arxiv.org/pdf/2201.05966.pdf)

[DataSet Link](https://unifiedskg.com/)

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

## Contributing

Our purpose is to make this repo even better. If you are interested in contributing, please refer to HERE for instructions in contribution.

## License

`Awesome-Prompt-Dataset` is released under the Apache 2.0 license.