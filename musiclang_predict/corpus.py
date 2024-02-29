import os
import glob

current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
midi_files = glob.glob(os.path.join(current_file_dir, "corpus", "*.mid"))


def list_corpus():
    return [os.path.basename(m).split('.')[0] for m in midi_files]


def get_midi_path_from_corpus(name):
    return os.path.join(current_file_dir, "corpus", f"{name}.mid")

