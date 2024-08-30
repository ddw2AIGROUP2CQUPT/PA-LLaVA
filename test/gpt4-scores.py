import base64
import requests
import pandas as pd
import json,jsonlines
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key = ""



import sqlite3

def create_db_and_table(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS caption
                 (id TEXT PRIMARY KEY, score TEXT)''')
    conn.commit()
    conn.close()


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_and_request(image_info,db_name):
    _id, label, test = image_info
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }
    prompt = "The following two sentences are descriptions of the same picture; give them a semantic similarity score out of a total of 10. Give your score in the format {'score': value} and give your explanation immediately afterward:\n1、"+test+"\n2、"+label

    payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    try:
        response = requests.post("https://pro.aiskt.com/v1/chat/completions", headers=headers, json=payload)
        content = response.json()
        caption = content['choices'][0]['message']['content']
    except:
        caption="<empty>"

    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO caption (id, score) VALUES (?, ?)',
              (_id, caption))
    conn.commit()
    conn.close()

    return {"id": _id, "score": caption}

# 主逻辑
if __name__ == "__main__":
    label_path = "label_path"
    test_path = "generate_path"
    with open(test_path,'r') as file:
        test_data = json.load(file)

    with open(label_path,'r') as file:
        label_data = json.load(file)

    key_all = label_data.keys()

    db_name = "path/data.db"
    create_db_and_table(db_name)
    
    # 构建需要处理的图像信息列表
    to_process = []
    for i in tqdm(key_all):
        _id = i
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # c.execute("SELECT * FROM caption WHERE id=? AND score = '<empty>'", (_id,))
        # result = c.fetchone()
        # conn.close()
        # if result:
        #     label = label_data[_id]
        #     test = test_data[_id]
        #     to_process.append((_id, label,test))
        # else:
        #     continue
        c.execute("SELECT * FROM caption WHERE id=?", (_id,))
        result = c.fetchone()
        conn.close()
        if result:
            continue
        else:
            label = label_data[_id]
            test = test_data[_id]
            to_process.append((_id, label,test))
            
    print(len(to_process))

    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_image = {executor.submit(encode_and_request, img_info,db_name): img_info for img_info in to_process}
        for future in tqdm(as_completed(future_to_image), total=len(to_process)):
            result = future.result()
