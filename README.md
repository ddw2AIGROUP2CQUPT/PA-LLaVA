# Pathology-LLaVA-(PCaption-0.5M dataset) 

We developed a domain-speciffc large language-vision assistant (PA-LLaVA) for pathology image understanding. Specifically, (1) we first construct a human pathology image-text dataset by cleaning the public medical image-text data for domainspecific alignment; (2) Using the proposed image-text data, we first train a pathology language-image pretraining (PLIP) model as the specialized visual encoder for pathology image, and then we developed scale-invariant connector to avoid the information loss caused by image scaling; (3) We adopt two-stage learning to train PA-LLaVA, first stage for domain alignment, and second stage for end to end visual question & answering (VQA) task.

## Updates：

**[24/08/30] The test code has been update!**  

**[24/08/20] Model weights have been updated in HuggingFace! [OpenFace-CQUPT/Pathology-LLaVA](https://huggingface.co/OpenFace-CQUPT/Pathology-LLaVA)**

## Architecture

![image](https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA/blob/main/Architecture.png)

## Checkpoint

The weights for PLIP and the weights for both the domain alignment and instruction fine-tuning phases of PA-LLaVA are disclosed in the HuggingFace(https://huggingface.co/OpenFace-CQUPT/Pathology-LLaVA).

## Human Pathology Image-Text data （PCaption-0.5M）

### Introduction
These public datasets contain substantial amounts of data unrelated to human pathology. To obtain the human pathology image-text data, we performed two cleaning processes on the raw data, as illustrated in the follow figture: (1) Removing nonpathological images. (2) Removing nonhuman pathology data. Additionally, we excluded image-text pairs with textual descriptions of fewer than 20 words. Ultimately, we obtained 518,413 image-text pairs (named "PCaption-0.5M" ) for the aligned training dataset.

Instruction fine-tuning phase we only cleaned PMC-VQA in the same way and obtained 15,788 question-answer pairs related to human pathology. Lastly, we combined PathVQA and Human pathology data obtained from PMC-VQA, thereby constructing a dataset of 35543 question-answer pairs data.

#### Data Cleaning Process

![image](https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA/blob/main/DataCleanProcess.png)

## Get the Dataset

### Step 1 Download the public datasets.
Here we only provide the download link for the public dataset and expose the image id index of our cleaned dataset on HuggingFace(https://huggingface.co/OpenFace-CQUPT/Pathology-LLaVA).
#### Domain Alignment Stage

PubMedVision-Alignment: [FreedomIntelligence/PubMedVision · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)

PMC-OA: [axiong/pmc_oa · Datasets at Hugging Face](https://huggingface.co/datasets/axiong/pmc_oa)

Quilt-1M: [Quilt-1M: One Million Image-Text Pairs for Histopathology (zenodo.org)](https://zenodo.org/records/8239942)


#### Instruction Tuning Stage

PathVQA: https://drive.google.com/drive/folders/1G2C2_FUCyYQKCkSeCRRiTTsLDvOAjFj5

PMC-VQA: [xmcmic/PMC-VQA · Datasets at Hugging Face](https://huggingface.co/datasets/xmcmic/PMC-VQA)


#### Categorical dataset for zero-sample testing

ICIAR 2018 BACH: https://iciar2018-challenge.grand-challenge.org/Download/

OSCC: https://data.mendeley.com/datasets/ftmp4cvtmb/1 

ColonPath : https://medfm2023.grand-challenge.org/datasets


### Step 2 Data processing.
First, use the image index of the clean dataset provided by us to extract the human pathological dataset, and then process it into the following format:
```
[
	{
		"image": ,
		"caption": 
	},
]
```

Finally, run dataformate.py to get the format needed to train the model.
```
python dataformat.py
```


## Training

We used xtuner as a training tool, so please go to xtuner official to complete the environment configuration [https://github.com/InternLM/xtuner]. Then add the xtuner_add/pallava file to the installed xtuner code with the following location structure.

![1725014395365](https://github.com/user-attachments/assets/ff37e700-c704-4a53-a216-f26b1aba5c05)



#### Domain Alignment
```
NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR= NODE_RANK=0 xtuner train pallava_domain_alignment.py --deepspeed deepspeed_zero2 --seed 1024
```

#### Instruction Tuning
```
NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR= NODE_RANK=0 xtuner train pallava_instruction_tuning.py --deepspeed deepspeed_zero2 --seed 1024
```

## Test

First, replace or add all the files in xtuner_add/tool_add into the tool file of the xtuner runtime file with the following file location structure:

![1725014736404](https://github.com/user-attachments/assets/87b7d87f-e980-4355-8777-dfedf0c54903)
### PathVQA
```
NPROC_PER_NODE=8 xtuner pathvqa meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder PLIP --llava ./instruction_tuning_weight_ft --prompt-template llama3_chat --data-path absolute_path/Path_VQA/path_vqa_test.json --work-dir absolute_path/logs/pathvqa --launcher pytorch --anyres-image
```

### PMCVQA
```
NPROC_PER_NODE=8 xtuner pmcvqa meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder PLIP --llava ./instruction_tuning_weight_ft --prompt-template llama3_chat --data-path absolute_path/PMC-VQA/pmc-vqa_test_clean_answer_abcd.json --work-dir absolute_path/logs/pmcvqa --launcher pytorch --anyres-image
```
### Zero-Shot
Here is an example with OSCC data.
#### Generate answer
```
NPROC_PER_NODE=8 xtuner zero_shot meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder PLIP --llava ./instruction_tuning_weight_ft --prompt-template llama3_chat --data-path absolute_path/OSCC/oscc.json --work-dir absolute_path/logs/oscc --launcher pytorch --anyres-image
```
#### Calculate score
```
python test/f1.py
```
### GPT4-Score
```
python python test/gpt4-scores.py
```


## Result
![image](https://github.com/user-attachments/assets/374027f5-bb3e-4a8e-ab25-d46aa328b908)

![zero_shot](https://github.com/user-attachments/assets/01d68f23-dda5-434d-96c5-746a9f031095)

## Citation
```
@misc{dai2024pallavalargelanguagevisionassistant,
      title={PA-LLaVA: A Large Language-Vision Assistant for Human Pathology Image Understanding}, 
      author={Dawei Dai and Yuanhui Zhang and Long Xu and Qianlan Yang and Xiaojing Shen and Shuyin Xia and Guoyin Wang},
      year={2024},
      eprint={2408.09530},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.09530}, 
}
```
## Contact
This repo is currently maintained by Dawei Dai (dw_dai@163.com) and his master's student Yuanhui Zhang (S230233056@stu.cqupt.edu.cn).


