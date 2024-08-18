import os
import re
import json,jsonlines
from glob import glob
from tqdm import tqdm
import random
from PIL import Image


result_jsonl = ""
prompt_txt = './pt_prompt.txt'
image_folder = ''
prompts = [line for line in open(prompt_txt, 'r').read().splitlines()]
read_json = ""

with open(read_json, 'r') as f:
    data = json.load(f)

instruct_data = []
for row in tqdm(data):
    _id = row['image'].split('.')[0]
    _image = os.path.join(image_folder,row['image'])
    image = Image.open(_image)
    width, height = image.size
    if width * height > 16777216:
        continue

    _instruct = random.sample(prompts, 1)[0]
    _instruct = _instruct + "\n<image>" if random.random() < 0.5 else "<image>\n" + _instruct
    _caption = row['caption']
    _conversations = []
    _conversations.append({"from": "human", "value": _instruct})
    _conversations.append({"from": "gpt", "value": _caption})
    instruct_data.append({
        "id": _id,
        "image": _image,
        "conversations": _conversations
    })
print(len(instruct_data))

with jsonlines.open(output_jsonl, 'w') as f:
    for item in instruct_data:
        f.write(item)
