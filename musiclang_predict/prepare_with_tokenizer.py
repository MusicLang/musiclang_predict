from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import os
import numpy as np
import pickle
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(files, output_tokenizer, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[MASK]"], vocab_size=vocab_size)
    tokenizer.train(files, trainer)
    tokenizer.save(os.path.join(output_tokenizer, "tokenizer.json"))
    return tokenizer


def encode_data(tokenizer, file):
    with open(file, 'r') as f:
        return tokenizer.encode(f.read()).ids


def join_directory(data_folder, output_directory, sep=';'):
    filenames = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    output_dataset = os.path.join(output_directory, 'data.txt')
    with open(output_dataset, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read() + f"{sep}\n")

    return output_dataset


def prepare_dataset_with_tokenizer(data_folder, output_directory, sep=';', vocab_size=1000):
    """
    Prepare a musiclang dataset for BPE encoding model
    It will first concatenate
    So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
    Will save train.bin, val.bin containing the ids, and meta.pkl containing the
    encoder and decoder and some other related info.

    Parameters
    -----------

    data_folder: str
        Folder on which we find the txt files for training (string representations of musiclang scores)
    output_directory: str
        Directory where we will create the data files and the metadata
    sep: str
        Character that separates two scores
    vocab_size: int
        Number of tokens allowed
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # download the tiny shakespeare dataset
    data_path = join_directory(data_folder, output_directory, sep=sep)
    # Concat in one file and put it here

    with open(data_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")


    # create the train and test splits
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode both to integers
    tokenizer = train_tokenizer([data_path], output_directory, vocab_size=vocab_size)
    train_ids = tokenizer.encode(train_data).ids
    val_ids = tokenizer.encode(val_data).ids
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(output_directory, 'train.bin'))
    val_ids.tofile(os.path.join(output_directory, 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size
    }

    with open(os.path.join(output_directory, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)