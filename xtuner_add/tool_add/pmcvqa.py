import argparse
import json
import math
import os
import os.path as osp
import re
import time
from PIL import Image
import torch
import tqdm
from huggingface_hub import snapshot_download
from mmengine import mkdir_or_exist
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from pallava.utils import process_anyres_image
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from pallava.flip.fflip import VisionModel,VisionConfig 
from xtuner.dataset.utils import (decode_base64_to_image, expand2square,
                                  get_bos_eos_token_ids)
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.tools.VQAEval import *
TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')



def parse_args():
    parser = argparse.ArgumentParser(description='VQA')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=1200,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument('--anyres-image', action='store_true', default=True, help='any res image')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()

    return args


@master_only
def master_print(msg):
    print(msg)




class EvalDataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.qid = list(data.keys())
        self.data = data
        self.img_dir = "image path/images"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        qid = self.qid[index]
        item = self.data[qid]
        question = item['question']
        image_name = item['image']
        image_path = os.path.join(self.img_dir, image_name)
        raw_image = Image.open(image_path)
        return {
            'question': question,
            'question_id': qid,
            'img': raw_image
        }

    @master_only
    def pre_answer(answer):
        answer = str(answer)
        answer = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            answer.lower(),
        ).replace(' \t', ' ')
        answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
        answer = answer.replace(' - ', '-')
        return answer
    
    def eval(self,res_path):
        correct = 0
        wrong = 0
        with open(res_path, 'r') as f:
            result = json.load(f)
        for row in result:
            pred = row['answer']
            match = re.match(r"([^:]+):", pred) #get the answer choice
            if match:
                pred = match.group(1)
            else:
                print(pred)
            pred = pred.lower()

            label = self.data[row['question_id']]['answer']
            label = label.lower()
            
            if pred == label:
                correct += 1
            else:
                wrong += 1
        print("correct: %d, wrong: %d" % (correct, wrong))
        print("acc: %.02f" % ((float(correct) / float(correct + wrong))*100))
    

        



def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': rank if world_size > 1 else 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    # build llm
    with LoadWoInit():
        llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)
    bos_token_id, _ = get_bos_eos_token_ids(tokenizer)
    master_print(f'Load LLM from {args.model_name_or_path}')

    llava_path = snapshot_download(
        repo_id=args.llava) if not osp.isdir(args.llava) else args.llava

    # build visual_encoder
    
    if 'visual_encoder' in os.listdir(llava_path):
        assert args.visual_encoder is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
    else:
        assert args.visual_encoder is not None, (
            'Please specify the `--visual-encoder`!')
        visual_encoder_path = args.visual_encoder

    with LoadWoInit():
        if visual_encoder_path == 'PLIP':
            vision_config = VisionConfig().from_json_file('path/plip/config.json')
            vision_checkpoint = torch.load('path/plip/plip_80w_196token.bin', map_location="cpu")
            visual_encoder = VisionModel.from_pretrained('openai/clip-vit-base-patch16', config = vision_config, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
            visual_encoder.load_state_dict(vision_checkpoint)
        else:
            visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])

        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)
    master_print(f'Load visual_encoder from {visual_encoder_path}')

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')

        with LoadWoInit():
            llm = PeftModel.from_pretrained(
                llm, adapter_path, offload_folder=args.offload_folder)

        master_print(f'Load LLM adapter from {args.llava}')

    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder, adapter_path, offload_folder=args.offload_folder)
        master_print(f'Load visual_encoder adapter from {args.llava}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    with LoadWoInit():
        projector = AutoModel.from_pretrained(
            projector_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],trust_remote_code=True)
    master_print(f'Load projector from {args.llava}')

    projector.cuda()
    projector.eval()

    visual_encoder.cuda()
    visual_encoder.eval()

    llm.eval()

    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    # work_dir
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        save_dir = args.work_dir
    else:
        # use config filename as default work_dir
        save_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.data_path))[0])
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    save_dir = osp.join(save_dir, timestamp)
    
    if rank == 0:
        mkdir_or_exist(osp.abspath(save_dir))
        print('=======================================================')
        print(f'Dataset path: {osp.abspath(args.data_path)}\n'
              f'Results will be saved to {osp.abspath(save_dir)}')
        print('=======================================================')

        args_path = osp.join(save_dir, 'args.json')
        with open(args_path, 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f, indent=2)

    results_json_path = osp.join(save_dir, 'pmcvqa_result.json')
    os.makedirs(osp.dirname(results_json_path), exist_ok=True)

    dataset = EvalDataset(args.data_path)
    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for i in tqdm.tqdm(per_rank_ids, desc=f'Rank {rank}'):
        data_sample = dataset[i]
        text = DEFAULT_IMAGE_TOKEN  + '\n' + data_sample['question']

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name=args.bot_name)
        else:
            prompt_text = text
        inputs = prompt_text
        image = data_sample['img'].convert('RGB')
        width, height = image.size

        if args.anyres_image:
            image = process_anyres_image(image, image_processor, None)
            image = image.cuda().to(visual_encoder.dtype)
        else:
            image = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)

        visual_outputs = visual_encoder(image, output_hidden_states=True)


        
        ori_pixel_embeds = visual_outputs.hidden_states[args.visual_select_layer][0, 1:].unsqueeze(0)
        dim = ori_pixel_embeds.shape[-1]
        patch_pixel_embeds = visual_outputs.hidden_states[args.visual_select_layer][1:, 1:].reshape(-1, dim).unsqueeze(0)
        pixel_values = projector(ori_pixel_embeds, patch_pixel_embeds)
        

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer.encode(chunk)
            else:
                cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2

        # TODO: Auto-detect whether to prepend a bos_token_id at the beginning.
        ids = bos_token_id.copy()

        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=llm, input_ids=ids, pixel_values=pixel_values)
        
        generate_output = llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)

        predict = tokenizer.decode(
            generate_output[0], skip_special_tokens=True).strip()

        new_data = {
            "question_id": data_sample["question_id"], 
            "answer": predict
        }
        results.append(new_data)

    results = collect_results(results, n_samples)
    if get_rank() == 0:
        with open(results_json_path, 'w') as f:
            json.dump(results, f)
        dataset.eval(results_json_path)

        
        

if __name__ == '__main__':

    main()
