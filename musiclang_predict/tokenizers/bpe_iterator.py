import re


class BPEIterator:
    r"""
    An iterable class to be used when training a musiclang tokenizer with BPE.

    It loads tokens text files to be used with the Hugging Face
    tokenizers library to build a vocabulary with BPE.

    It splits the tokens into different sequences using all the control tokens as separator
    Eg : CHORD_CHANGE INSTRUMENT_NAME__piano INSTRUMENT_PART__0 NOTE_TYPE_s NOTE_VAL__0 NOTE_OCTAVE__0 ... will be splitted as
    ['CHORD_CHANGE', 'INSTRUMENT_NAME__piano', 'INSTRUMENT_PART__0', 'NOTE_TYPE_s NOTE_VAL__0 NOTE_OCTAVE__0 ...']
    separating note tokens from control tokens.

    """

    def __init__(self, tokenizer, files_paths, control_tokens=[]) -> None:
        self.tokenizer = tokenizer
        self.files_paths = files_paths
        self.control_tokens = control_tokens
        self.__iter_count = 0

    def load_file(self, path):
        """
        Load a MIDI file and convert it to its byte representation.

        :param path: path to the file to load.
        :return: the byte representation of the file.
        """
        with open(path, 'r') as f:
            text = f.read()

        # list of str (bytes)
        bytes_ = self.tokenizer.tokens_to_bytes(text)
        bytes_ = bytes_[:8000]

        # Split
        split_pattern = '|'.join([re.escape(token) for token in self.control_tokens])
        bytes_ = re.split(f'({split_pattern})', bytes_)
        bytes_ = [b for b in bytes_ if len(b) > 0]
        return bytes_

    def __len__(self):
        """
        Return the number of files in the training corpus.

        :return: number of files in the training corpus.
        """
        return len(self.files_paths)

    def __getitem__(self, idx: int):
        """
        Convert the ``idx``th file to its byte representation.

        :param idx: idx of the file to convert.
        :return: byte representation of the file.
        """
        return self.load_file(self.files_paths[idx])

    def __iter__(self):  # noqa:D105
        return self

    def __next__(self) :  # noqa:D105
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]

    def __str__(self):
        """
        Return the ``str`` representation of the iterator.

        :return: string description.
        """
        return f"{self.tokenizer} - {len(self)} files"

