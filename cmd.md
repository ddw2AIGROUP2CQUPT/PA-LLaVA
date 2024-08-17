### domain alignment 
NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR= NODE_RANK=0 xtuner train pallava_domain_alignment.py --deepspeed deepspeed_zero2 --seed 1024
### instruction tuning
NPROC_PER_NODE=8 NNODES=2 PORT=12345 ADDR= NODE_RANK=0 xtuner train pallava_instruction_tuning.py --deepspeed deepspeed_zero2 --seed 1024