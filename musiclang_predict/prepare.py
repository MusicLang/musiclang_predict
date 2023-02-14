import os
import pickle
import numpy as np



def join_directory(data_folder, output_directory, sep=';'):
    filenames = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    output_dataset = os.path.join(output_directory, 'data.txt')
    if not os.path.exists(output_dataset):
        with open(output_dataset, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read() + f"{sep}\n")

    return output_dataset

def prepare_dataset(data_folder, output_directory, sep=';'):
    """
    Prepare a musiclang dataset for character-level language modeling.
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
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # download the tiny shakespeare dataset
    data_path = join_directory(data_folder, output_directory, sep=sep)
    # Concat in one file and put it here

    with open(data_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(output_directory, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)