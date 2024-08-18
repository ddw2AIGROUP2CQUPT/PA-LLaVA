# PA-LLaVA: A Large Language-Vision Assistant for Human Pathology Image Understanding

we developed a domain-speciffc large language-vision assistant (PA-LLaVA) for pathology image understanding.

## Architecture

 [_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=3916901031848625157&skey=@crypt_24346367_9184997c3f15be5a959433ddab101a36&mmweb_appid=wx_webfilehelper.jfif](_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=3916901031848625157&skey=@crypt_24346367_9184997c3f15be5a959433ddab101a36&mmweb_appid=wx_webfilehelper.jfif) 

## Model

领域对齐和指令微调两个阶段的权重均公开于HuggingFace.[ddw2openface/PA-LLaVA · Hugging Face](https://huggingface.co/ddw2openface/PA-LLaVA)

## Data

Here we only provide the download link for the public dataset and expose the image id index of our cleaned dataset on HuggIngface.

#### Domain Alignment Stage

PubMedVision-Alignment: [FreedomIntelligence/PubMedVision · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)

PMC-OA: [axiong/pmc_oa · Datasets at Hugging Face](https://huggingface.co/datasets/axiong/pmc_oa)

Quilt-1M: [Quilt-1M: One Million Image-Text Pairs for Histopathology (zenodo.org)](https://zenodo.org/records/8239942)

#### Instruction Tuning Stage

PathVQA: https://link.zhihu.com/?target=https%3A//drive.google.com/drive/folders/1G2C2_FUCyYQKCkSeCRRiTTsLDvOAjFj5

PMC-VQA: [xmcmic/PMC-VQA · Datasets at Hugging Face](https://huggingface.co/datasets/xmcmic/PMC-VQA)

#### Categorical dataset for zero-sample testing

ICIAR 2018 BACH: https://iciar2018-challenge.grand-challenge.org/Download/

OSCC: https://data.mendeley.com/datasets/ftmp4cvtmb/1 

ColonPath : https://medfm2023.grand-challenge.org/datasets

## Training

We used xtuner as a training tool, so please go to xtuner official to complete the environment configuration [https://github.com/InternLM/xtuner]. Then place the pallava folder under the xtuner_add folder into the xtuner folder.

Domain Alignment

NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR= NODE_RANK=0 xtuner train pallava_domain_alignment.py --deepspeed deepspeed_zero2 --seed 1024

Instruction Tuning

NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR= NODE_RANK=0 xtuner train pallava_instruction_tuning.py --deepspeed deepspeed_zero2 --seed 1024
