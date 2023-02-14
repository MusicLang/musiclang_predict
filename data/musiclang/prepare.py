import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
filepath = "/content/drive/MyDrive/training_data"
filepath = "/Users/floriangardin/code/music/musiclang2/locals/dataset_lmd"
output_filepath = "/Users/floriangardin/code/music/musiclang2/locals/musiclang.txt"

# download the tiny shakespeare dataset
input_file_path = filepath
filenames = [os.path.join(filepath, f) for f in os.listdir(input_file_path)]

# Concat in one file and put it here
if not os.path.exists(output_filepath):
    with open(output_filepath, 'w') as outfile:
       for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read() + ";\n")

with open(output_filepath, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
