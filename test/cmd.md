## All Path need replace by yourself

## PathVQA  path:/xtuner/xtuner/tools/pathvqa.py
NPROC_PER_NODE=8 xtuner pathvqa meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder PLIP --llava ./instruction_tuning_weight_ft --prompt-template llama3_chat --data-path absolute_path/Path_VQA/path_vqa_test.json --work-dir absolute_path/logs/pathvqa --launcher pytorch --anyres-image

## PMCVQA  path:/xtuner/xtuner/tools/pmcvqa.py
NPROC_PER_NODE=8 xtuner pmcvqa meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder PLIP --llava ./instruction_tuning_weight_ft --prompt-template llama3_chat --data-path absolute_path/PMC-VQA/pmc-vqa_test_clean_answer_abcd.json --work-dir absolute_path/logs/pmcvqa --launcher pytorch --anyres-image

## Zero-Shot  path:/xtuner/xtuner/tools/zero_shot.py
### Using OSCC as an example
### 1
NPROC_PER_NODE=8 xtuner zero_shot meta-llama/Meta-Llama-3-8B-Instruct --visual-encoder PLIP --llava ./instruction_tuning_weight_ft --prompt-template llama3_chat --data-path absolute_path/OSCC/oscc.json --work-dir absolute_path/logs/oscc --launcher pytorch --anyres-image
### 2
python f1.py



