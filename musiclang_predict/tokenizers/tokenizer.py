import copy

from musiclang import Score, Chord, Note, Melody, Tonality
import os
import tempfile
import os

import json
from fractions import Fraction as frac
import joblib
import numpy as np
from multiprocessing import Pool
import functools
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import gc
from fractions import Fraction as frac

from tokenizers import Tokenizer, models, trainers

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer


from musiclang_predict.tokenizers.bpe_iterator import BPEIterator

NOTE_DURATION_MAX_DENOMINATOR = 8
CHORD_DURATION_MAX_DENOMINATOR = 4
BASE_CHAR_ID = 33
TOKENIZER_CONFIG_BASE = {
  "model_max_len": 4096,
  "model_max_length": 4096,
  "pad_token": "[PAD]",
  "pad_token_id": 3,
  "eos_token": "[END]",
  "eos_token_id": 45,
  "bos_token": "[START]",
  "bos_token_id": 5,
  "mask_token": "[MASK]",
  "mask_token_id": 4,
  "unk_token": "[UNK]",
  "unk_token_id": 0,
  "cls_token": "[CLS]",
  "cls_token_id": 1,
  "sep_token": "[SEP]",
  "seo_token_id": 2,
  "add_prefix_space": False,
  "tokenizer_class": "PreTrainedTokenizer",
  "type": "Split"
}


default_options = {
    'chord_change_token': True,
    'melody_end_token': True,
    'chord_duration_token': True,
    'density_token': True,
    'chord_extension_token': True,
    'next_chord_token': True,
    'next_chord_duration_token': True,
    'will_end_token': True,
    'dissonance_token': True,
    'amplitude_token': True,
    'average_octave_token': True,
    'voice_token': False,
    'random_instrument_permutation': True,
    'end_token': True,
    'silence_continuation_eco': True
}
class MusicLangTokenizer:
    """
    Convert a score into a list of tokens
    """

    SCORE_START = 'SCORE_START'
    UNKNOWN = 'UNKNOWN'

    CHORD_DEGREE = 'CHORD_DEGREE'
    TONALITY_DEGREE = 'TONALITY_DEGREE'
    TONALITY_MODE = 'TONALITY_MODE'
    CHORD_OCTAVE = 'CHORD_OCTAVE'
    CHORD_DURATION_NUM = 'CHORD_DURATION_NUM'
    CHORD_DURATION_DEN = 'CHORD_DURATION_DEN'

    CHORD_EXTENSION = 'CHORD_EXTENSION'

    NEXT_CHORD_DEGREE = 'NEXT_CHORD_DEGREE'
    NEXT_TONALITY_DEGREE = 'NEXT_TONALITY_DEGREE'
    NEXT_TONALITY_MODE = 'NEXT_TONALITY_MODE'
    NEXT_CHORD_OCTAVE = 'NEXT_CHORD_OCTAVE'
    NEXT_CHORD_EXTENSION = 'NEXT_CHORD_EXTENSION'
    NEXT_CHORD_DURATION_NUM = 'NEXT_CHORD_DURATION_NUM'
    NEXT_CHORD_DURATION_DEN = 'NEXT_CHORD_DURATION_DEN'

    CHORD_CHANGE = 'CHORD_CHANGE'
    MELODY_END = 'MELODY_END'
    WILL_END = 'WILL_END'
    DISSONANCE = 'DISSONANCE'
    AMPLITUDE = 'AMPLITUDE'

    INSTRUMENT_NAME = 'INSTRUMENT_NAME'
    INSTRUMENT_PART = 'INSTRUMENT_PART'
    DENSITY = 'DENSITY'
    AVERAGE_OCTAVE = 'AVERAGE_OCTAVE'

    NOTE_TYPE = 'NOTE_TYPE'
    NOTE_VAL = 'NOTE_VAL'
    NOTE_OCTAVE = 'NOTE_OCTAVE'
    NOTE_AMP = 'NOTE_AMP'
    NOTE_DURATION_NUM = 'NOTE_DURATION_NUM'
    NOTE_DURATION_DEN = 'NOTE_DURATION_DEN'
    END = 'END'

    def __init__(self, tokenizer_path=None, options=None, hub_tokenizer_path='tokenizer-base.json'):
        self.dict = {}
        self.tokenizer = None
        self.denoms = [i for i in range(1, NOTE_DURATION_MAX_DENOMINATOR + 1)]
        if tokenizer_path is None:
            import warnings
            warnings.warn("No tokenizer_path provided. Using a new tokenizer. You probably should train it using 'train_tokenizer_from_token_files' method.")
        else:
            try:
                with open(tokenizer_path, 'r') as f:
                    self.dict = json.load(f)
            except Exception as e:
                tokenizer_path_hub = hf_hub_download(repo_id=tokenizer_path, filename=hub_tokenizer_path)
                with open(tokenizer_path_hub, 'r') as f:
                    self.dict = json.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.denoms = self.get_avalaible_denominators()
        # Replace str to int for keys of id_to_token
        if options is not None:
            self.dict['options'] = options
        elif 'options' not in self.dict:
            self.dict['options'] = default_options

    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab_size)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.tokenizer.vocab[item]
        elif isinstance(item, int):
            return self.tokenizer.decode(item)
        else:
            raise ValueError(f"Invalid type {type(item)} for item {item}")

    def tokenize_midi_file(self, midi_file, quantization=8, fast_chord_inference=True, chord_range=None, interceptors=None):
        """
        Tokenize a single midi file and returns list of tokens
        Parameters
        ----------
        midi_file
        quantization
        fast_chord_inference
        chord_range

        Returns
        -------

        """
        score = Score.from_midi(midi_file, quantization=quantization, fast_chord_inference=fast_chord_inference, chord_range=chord_range)
        if interceptors is not None:
            for interceptor in interceptors:
                score = interceptor(score)
        tokens = self.tokenize(score)
        return tokens

    def tokenize_midi_files(self, midi_files, quantization=8, fast_chord_inference=True, chord_range=None):
        all_tokens = []
        for midi_file in midi_files:
            tokens = self.tokenize_midi_file(midi_file, quantization=quantization, fast_chord_inference=fast_chord_inference, chord_range=chord_range)
            all_tokens.append(tokens)
        return all_tokens


    def calculate_tokens_to_ids_dict(self, token_files):

        unique_tokens = set()
        for token_file in token_files:
            with open(token_file, 'r') as f:
                tokens = f.read().split('\n')
                unique_tokens.update(tokens)
        unique_tokens = list(sorted(list(unique_tokens)))
        dict = {token: idx for idx, token in enumerate(unique_tokens)}
        inv_dict = {idx: token for idx, token in enumerate(unique_tokens)}
        self.dict = {'options': self.dict['options']}


    def tokenize_from_file(self, path):
        """
        Tokenize a file and returns a list of tokens
        Parameters
        ----------
        path: str, path to file

        Returns
        -------
        tokens: List[str], list of tokens
        """
        with open(path, 'r') as f:
            tokens = f.read().split()
        return tokens

    def tokenize_sequence(self, seq):
        """
        Tokenize a sequence of tokens (A pandas dataframe with appropriate columns
        Parameters
        ----------
        seq: pd.DataFrame

        Returns
        -------
        tokens: str
        """

        score = Score.from_sequence(seq)
        return self.tokenize(score)



    def train_tokenizer_from_token_files(self, token_files, output=None, hub_output=None, **kwargs):
        """
        Train a tokenizer from a list of token files.
        It will also save a tokenizer-base.json file with the options used to train the tokenizer
        (needed for tokenization from musiclang language)

        Make sure you have logged in to huggingface using `huggingface-cli login` before training and pushing to the hub
        Parameters
        ----------
        token_files:
        output_tokenizer: Path to save the tokenizer
        output: Path to save the tokenizer and the config file (Either output or hub_output must be not None)
        hub_output: Path to save the tokenizer and the config file in the huggingface hub

        Returns
        -------
        tokenizer: Tokenizer
        """


        def train_tokenizer(data_files, vocab_size=30_000,
                            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[START]", "[END]"]):
            # Create a tokenizer
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            # Pre-tokenizer to split the text into words
            tokenizer.pre_tokenizer = WhitespaceSplit()
            # tokenizer.pre_tokenizer = PreTokenizer.custom("whitespace_split", "regex", pattern=r"\s+")
            # Special tokens
            tokenizer.add_special_tokens(special_tokens)
            # Trainer
            trainer = WordLevelTrainer(
                # vocab_size=vocab_size,
                special_tokens=special_tokens,
            )

            # Train the tokenizer
            tokenizer.train(files=data_files, trainer=trainer)
            return tokenizer

        # Example usage
        data_files = token_files  # Replace with the path to your text file
        tokenizer = train_tokenizer(data_files)
        # Save the tokenizer to a temp directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save(os.path.join(tmpdirname, "tokenizer.json"))

            # Save TOKENIZER_BASE as tokenizer_config.json in directory
            options = copy.deepcopy(TOKENIZER_CONFIG_BASE)
            for key, value in kwargs.items():
                options[key] = value

            with open(os.path.join(tmpdirname, 'tokenizer_config.json'), 'w') as f:
                json.dump(options, f, indent=4)
            # Reload tokenizer to push it or save it
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)
            if output is not None:
                tokenizer.save_pretrained(output)
                # Save base options
                with open(os.path.join(output, 'tokenizer-base.json'), 'w') as f:
                    json.dump({"options": self.dict['options']}, f, indent=4)
            if hub_output is not None:
                tokenizer.push_to_hub(hub_output)
                # Push base options to hub (1. save to a tempfile, 2. then use hf api to push to hub)
                with tempfile.TemporaryDirectory() as tmpdirname:
                    with open(os.path.join(tmpdirname, 'tokenizer-base.json'), 'w') as f:
                        json.dump({"options":  self.dict['options']}, f, indent=4)
                    from huggingface_hub import HfApi
                    hf_api = HfApi()
                    hf_api.upload_file(
                        path_or_fileobj=os.path.join(tmpdirname, 'tokenizer-base.json'),
                        path_in_repo="tokenizer-base.json",
                        repo_id=hub_output,
                        repo_type="model",
                    )

            if output is None and hub_output is None:
                # Raise ONLY a warning
                print("WARNING: hub_output is None, tokenizer not pushed to hub")

        return tokenizer

    def __call__(self, score, **kwargs):
        tokens = " ".join(self.tokenize(score))
        return self.tokenizer(tokens, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def tokenize_chords(self, score):
        return self.tokenize(score, only_chords=True)

    def ids_to_tokens(self, ids):
        """
        Convert a list of token ids to a list of tokens
        Parameters
        ----------
        ids

        Returns
        -------

        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_score(self, ids):
        """
        Convert a list of token ids to a score
        Parameters
        ----------
        ids

        Returns
        -------

        """
        tokens = self.ids_to_tokens(ids)
        return self.untokenize(tokens)

    def file_ids_to_score(self, path):
        """
        Convert a file with ids to a score
        Parameters
        ----------
        path: Path to numpy array with ids

        Returns
        -------

        """
        tokens_ids = joblib.load(path)
        tokens = self.ids_to_tokens(tokens_ids)

        return self.untokenize(tokens)


    def file_tokens_to_score(self, path):
        """
        Convert a file with tokens to a score
        Parameters
        ----------
        path: Path to file with tokens

        Returns
        -------

        """
        with open(path, 'r') as f:
            tokens = f.read().split('\n')
        return self.untokenize(tokens)

    def tokenize_to_ids(self, score, include_end=True):
        """
        Tokenize a score and returns a list of token ids
        Parameters
        ----------
        score: MusicLang.Score, score to tokenize
        include_end: bool, if True add the END token at the end of the list (default=True)

        Returns
        -------
        ids: List[int], list of token ids

        """
        tokens = self.tokenize(score)
        ids = self.tokens_to_ids(tokens)
        if not include_end:
            ids = ids[:-1]
        return ids

    def tokens_to_ids(self, tokens):
        """
        Convert a list of tokens to a list of token ids
        Parameters
        ----------
        tokens

        Returns
        -------

        """
        if self.tokenizer is None:
            raise ValueError("No tokens to ids available. You must train the tokenizer first using train_tokenizer_from_token_files method.")
        res_with_tokenizer = self.tokenizer(" ".join(tokens))

        return res_with_tokenizer['input_ids']


    def train_bpe(self, files_paths, output_dir=None, hub_path=None, vocab_size=30_000, type='sentence_piece'):
        """
        :param files_paths: list of str
        Tokens text files paths
        :param output_dir: Output of the BPE model, if None, a temporary directory will be created
        :param hub_path: str
        Path to the hub where to push the tokenizer
        :param vocab_size: int
        Number of tokens in the BPE model
        :param type: str (sentence_piece or bpe)
        :return:
        """


        from tokenizers import SentencePieceBPETokenizer
        from tokenizers import normalizers
        # Initialize the BPEIterator with your custom tokenizer and list of file paths
        control_tokens = self.get_control_tokens_bytes()
        bpe_iterator = BPEIterator(self, files_paths, control_tokens=control_tokens)

        # Initialize the Hugging Face tokenizer with a BPE model
        if type == 'sentence_piece':
            tokenizer = SentencePieceBPETokenizer(add_prefix_space = False)
            tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            tokenizer.normalizer = normalizers.Sequence([])
            tokenizer.train_from_iterator(bpe_iterator, special_tokens=['<unk>', '<pad>', '<bos>', '<eos>', '<mask>'], vocab_size=vocab_size, show_progress=True)
        elif type == 'bpe':
            tokenizer = Tokenizer(models.BPE())
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=True, max_token_length=32)
            tokenizer.train_from_iterator(bpe_iterator, trainer=trainer)
        else:
            raise ValueError(f"Invalid type {type}, must be 'sentence_piece' or 'bpe'")

        # Save the trained tokenizer
        file = "tokenizer.json"
        delete_temp_dir = False
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            delete_temp_dir = True

        tokenizer_file = os.path.join(output_dir, file)
        tokenizer.save(tokenizer_file)
        tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer.model.save(output_dir)

        if hub_path is not None:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, model_max_length=self.tokenizer.model_max_length)
            tokenizer.push_to_hub(hub_path)

        # Delete the temporary directory
        if delete_temp_dir:
            import shutil
            shutil.rmtree(output_dir)

        # Push to hub

        return tokenizer


    def tokens_to_bytes(self, tokens, as_one_str=True):
        if not isinstance(tokens, str):
            tokens = " ".join(tokens)
        ids = self.tokenizer(tokens)['input_ids']
        bytes_ids = [chr(i + BASE_CHAR_ID) for i in ids]
        if as_one_str:
            return ''.join(bytes_ids)
        return bytes_ids


    def get_control_tokens(self):
        """
        Get all the tokens that are control tokens and that should not be part of BPE merges
        :return:
        """

        # Control tokens are all tokens that are not note tokens
        control_tokens = [voc for voc in self.tokenizer.vocab.keys() if not voc.startswith('NOTE_')]
        return control_tokens

    def get_control_tokens_bytes(self):
        """
        Get all the tokens that looks like control tokens
        :return:
        """
        # Control tokens are all tokens that are not note tokens
        control_tokens = [chr(i + BASE_CHAR_ID) for i in self.tokenizer.vocab.values() if not self.tokenizer.decode(i).startswith('NOTE_')]
        return control_tokens

    def bytes_to_tokens(self, bytes, to_str=True):
        ids = [ord(b) - BASE_CHAR_ID for b in bytes]
        ids = [id for id in ids if id >= 0]
        result = self.ids_to_tokens(ids)
        if to_str:
            return " ".join(result)
        return result

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.dict, f, indent=4)

    def tokenize_to_files_single(self, midi_files, output_dir, quantization=8,
                          fast_chord_inference=True, chord_range=None, sep="\n",
                          allow_error=True
                          ):
        """
        Tokenize a list of midi files and save the tokens to files
        Parameters
        ----------
        midi_files
        output_dir
        quantization
        fast_chord_inference
        chord_range

        Returns
        -------

        """
        for midi_file in tqdm(midi_files, 'Tokenizing files'):
            try:
                tokens = self.tokenize_midi_file(midi_file, quantization=quantization, fast_chord_inference=fast_chord_inference, chord_range=chord_range)
                filename = os.path.basename(midi_file).replace('.mid', '.txt')
                output_file = os.path.join(output_dir, filename)
                with open(output_file, 'w') as f:
                    f.write(sep.join(tokens))
            except Exception as e:
                if allow_error:
                    print(f"Error while tokenizing {midi_file}: {e}")
                else:
                    raise e


    def worker(self, args):
        gc.collect()  # Force garbage collection at the start of the worker
        midi_file, output_dir, quantization, fast_chord_inference, interceptors, chord_range, sep, allow_error = args
        try:
            tokens = self.tokenize_midi_file(midi_file, quantization=quantization,
                                             fast_chord_inference=fast_chord_inference,
                                             chord_range=chord_range,
                                             interceptors=interceptors
                                             )
            filename = os.path.basename(midi_file).replace('.mid', '.txt')
            output_file = os.path.join(output_dir, filename)
            with open(output_file, 'w') as f:
                f.write(sep.join(tokens))
        except Exception as e:
            if allow_error:
                print(f"Error while tokenizing {midi_file}: {e}")
            else:
                raise e
        finally:
            gc.collect()  # Force garbage collection at the end of the worker

    def tokenize_to_files(self, midi_files, output_dir, quantization=8,
                          fast_chord_inference=True, chord_range=None, sep="\n",
                          allow_error=True, interceptors=None, num_processes=4
                          ):
        args = [(midi_file, output_dir, quantization, fast_chord_inference, interceptors, chord_range, sep, allow_error)
                for midi_file in midi_files]

        with Pool(num_processes, maxtasksperchild=1) as pool:
            for _ in tqdm(pool.imap_unordered(self.worker, args), total=len(midi_files), desc='Tokenizing files'):
                pass


    def density_to_density_str(self, density):
        # densities = {'low': (0, 0.55), 'medium': (0.55, 1.5), 'high': (1, 2), 'very_high': (2, 10)}
        if density < 0.55:
            return 'low'
        elif density < 1.9:
            return 'medium'
        elif density < 3:
            return 'high'
        else:
            return 'very_high'

    def tokenize_to_bytes(self, score, as_one_str=True):
        tokens = self.tokenize(score)
        return self.tokens_to_bytes(tokens, as_one_str=as_one_str)

    def untokenize_from_bytes(self, bytes):
        tokens = self.bytes_to_tokens(bytes, to_str=False)
        return self.untokenize(tokens)


    def tokenize(self, score, only_chords=False, for_prompt=False):
        if self.dict['options'].get('random_instrument_permutation', False):
            import random
            instruments = score.instruments
            random.shuffle(instruments)
            sort_dict = {ins: idx for idx, ins in enumerate(instruments)}

        if isinstance(score, Chord):
            score = Score(chords=[score])
        tokens = []
        for idx, chord in enumerate(score):
            densities = chord.to_score().extract_densities()
            octave_means = chord.to_score().extract_mean_octaves()
            amplitude_means = chord.to_score().extract_mean_amplitudes()

            tokens_chord = self.tokenize_chord(chord, only_chords=only_chords)
            tokens += tokens_chord
            if self.dict['options'].get('next_chord_token', False) and not only_chords:
                # Last element
                if idx < len(score.chords) - 1:
                    next_chord = score.chords[idx + 1]
                    tokens_next_chord = self.tokenize_next_chord(next_chord)
                    tokens += tokens_next_chord
                else:
                    if self.dict['options'].get('will_end_token', True):
                        tokens += [self.WILL_END]

            if not only_chords:

                if self.dict['options'].get('random_instrument_permutation', False):
                    items = sorted(chord.score.items(), key=lambda x: sort_dict[x[0]])
                else:
                    items = chord.score.items()

                for ins, melody in items:
                    ins_name, ins_part = ins.split('__')
                    tokens_ins_name = self.INSTRUMENT_NAME + '__' + ins_name
                    tokens.append(tokens_ins_name)
                    if self.dict['options'].get('voice_token', True):
                        tokens_ins_part = self.INSTRUMENT_PART + '__' + ins_part
                        tokens.append(tokens_ins_part)
                    if self.dict['options'].get('density_token', False):
                        instrument_density = densities[ins]
                        density_str = self.density_to_density_str(instrument_density)
                        tokens_density = self.DENSITY + '__' + density_str
                        tokens += [tokens_density]
                    if self.dict['options'].get('average_octave_token', False):
                        tokens_average_octave = self.AVERAGE_OCTAVE + '__' + str(octave_means[ins])
                        tokens += [tokens_average_octave]
                    if self.dict['options'].get('amplitude_token', False):
                        tokens_amplitude = self.AMPLITUDE + '__' + str(amplitude_means[ins])
                        tokens += [tokens_amplitude]
                    for note in melody:
                        tokens_note = self.tokenize_note(note)
                        tokens += tokens_note
                    if self.dict['options'].get('melody_end_token', False):
                        tokens.append(self.MELODY_END)

        if self.dict['options'].get('end_token', True):
            tokens.append(self.END)

        return tokens

    def tokenize_chord_duration(self, chord_duration):
        token_chord_duration_num = self.CHORD_DURATION_NUM + '__' + str(chord_duration.numerator)
        token_chord_duration_den = self.CHORD_DURATION_DEN + '__' + str(chord_duration.denominator)
        return [token_chord_duration_num, token_chord_duration_den]


    def get_avalaible_denominators(self):
        if self.tokenizer is not None:
            words = self.tokenizer.vocab
            self.denoms = [int(w.split('__')[1]) for w in words if 'NOTE_DURATION_DEN' in w]
        return self.denoms

    def tokenize_note(self, note):
        note_type = self.NOTE_TYPE + '__' + note.type
        note_degree = self.NOTE_VAL + '__' + str(note.val)
        note_octave = self.NOTE_OCTAVE + '__' + str(note.octave)
        note_amp = self.NOTE_AMP + '__' + note.amp_figure

        # Limit denominator of duration to 4
        note_duration = frac(note.duration).limit_denominator(NOTE_DURATION_MAX_DENOMINATOR)
        available_denominators = self.denoms
        note_duration_den = min(available_denominators, key=lambda x: abs(note_duration.denominator - x))
        if note_duration.numerator == 0:
            return []

        # Find best approximation using denominator
        note_duration = frac(note_duration.numerator, note_duration_den)
        note_duration_num = self.NOTE_DURATION_NUM + '__' + str(note_duration.numerator)
        note_duration_den = self.NOTE_DURATION_DEN + '__' + str(note_duration.denominator)
        # Create the list
        tokens = [note_type]
        if (note.type not in ['r', 'l']) or not self.dict['options'].get('silence_continuation_eco', False):
            tokens += [note_degree, note_octave, note_amp]

        tokens += [note_duration_num, note_duration_den]

        return tokens

    def tokenize_chord(self, chord, only_chords=False):
        tokens = []
        if self.dict['options']['chord_change_token']:
            tokens.append(self.CHORD_CHANGE)

        chord_degree = self.CHORD_DEGREE + '__' + str(chord.element)
        chord_octave = self.CHORD_OCTAVE + '__' + str(chord.full_octave)
        chord_extension = self.CHORD_EXTENSION + '__' + str(chord.extension)
        tonality_degree = self.TONALITY_DEGREE + '__' + str(chord.tonality.degree)
        tonality_mode = self.TONALITY_MODE + '__' + chord.tonality.mode

        # Create the list
        tokens += [chord_degree, tonality_degree, tonality_mode, chord_octave]

        if self.dict['options'].get('chord_extension_token', False):
            tokens += [chord_extension]

        if self.dict['options'].get('chord_duration_token', False) and not only_chords:
            chord_duration = frac(chord.duration).limit_denominator(CHORD_DURATION_MAX_DENOMINATOR)
            chord_duration_num = self.CHORD_DURATION_NUM + '__' + str(chord_duration.numerator)
            chord_duration_den = self.CHORD_DURATION_DEN + '__' + str(chord_duration.denominator)
            tokens += [chord_duration_num, chord_duration_den]

        return tokens

    def tokenize_next_chord(self, chord):
        tokens = []

        chord_degree = self.NEXT_CHORD_DEGREE + '__' + str(chord.element)
        chord_octave = self.NEXT_CHORD_OCTAVE + '__' + str(chord.octave)
        tonality_degree = self.NEXT_TONALITY_DEGREE + '__' + str(chord.tonality.degree)
        tonality_mode = self.NEXT_TONALITY_MODE + '__' + chord.tonality.mode

        tokens += [chord_degree, tonality_degree, tonality_mode, chord_octave]

        if self.dict['options'].get('next_chord_duration_token', True):
            chord_duration = frac(chord.duration).limit_denominator(CHORD_DURATION_MAX_DENOMINATOR)
            chord_duration_num = self.NEXT_CHORD_DURATION_NUM + '__' + str(chord_duration.numerator)
            chord_duration_den = self.NEXT_CHORD_DURATION_DEN + '__' + str(chord_duration.denominator)
            tokens += [chord_duration_num, chord_duration_den]

        if self.dict['options']['chord_extension_token']:
            chord_extension = self.NEXT_CHORD_EXTENSION + '__' + str(chord.extension)
            tokens += [chord_extension]

        return tokens





    def untokenize(self, tokens):
        if isinstance(tokens, str):
            # Split by \n or whitespace
            tokens = tokens.split()

        score = Score()
        current_chord = None
        current_melody = None
        current_instrument_name = None
        current_instrument_part = None

        current_chord_duration_num = None
        current_chord_duration_den = None
        chord_duration = None
        current_melody_duration = 0
        note_duration_num = 0
        note_duration_den = 0
        note_val = 0
        note_type = 'r'
        note_octave = 0
        note_amp = 'mf'

        current_instrument_idx = {}

        for token in tokens:
            # Split token into key and value
            if token in [self.END, self.CHORD_CHANGE, self.WILL_END, self.SCORE_START]:
                continue

            if token == self.MELODY_END:
                # Check if melody duration is equal to chord duration, else add a rest
                if current_melody_duration < chord_duration:
                    delta = chord_duration - current_melody_duration
                    note_duration_num = int(delta.numerator)
                    note_duration_den = int(delta.denominator)
                    note_duration = frac(note_duration_num, note_duration_den)
                    if note_duration > 0:
                        note = Note(type='r', val=0, octave=0, duration=note_duration)
                        current_melody.notes.append(note)
                # Then continue
                current_chord.score[current_instrument_name + '__' + current_instrument_part] = current_melody
                continue

            try:
                key, value = token.split('__')
            except:
                raise ValueError(f"Invalid token {token}")

            if key == self.CHORD_DEGREE:
                if current_chord is not None:
                    score.chords.append(current_chord)
                current_chord = Chord(element=int(value), tonality=Tonality(0, 'M'))
                current_instrument_idx = {}

            elif key == self.CHORD_OCTAVE:
                current_chord.octave = int(value)

            elif key == self.CHORD_DURATION_NUM:
                current_chord_duration_num = int(value)

            elif key == self.CHORD_DURATION_DEN:
                current_chord_duration_den = int(value)
                chord_duration = frac(current_chord_duration_num, current_chord_duration_den)

            elif key == self.TONALITY_DEGREE:
                current_chord.tonality.degree = int(value)  # Assuming Tonality can be constructed from a string

            elif key == self.TONALITY_MODE:
                current_chord.tonality.mode = value

            elif key == self.CHORD_EXTENSION:
                current_chord.extension = value

            elif key == self.INSTRUMENT_NAME:
                current_instrument_name = value
                current_melody_duration = 0
                note_duration_num = 0
                note_duration_den = 0
                current_melody = Melody(notes=[])

                if current_instrument_name not in current_instrument_idx:
                    current_instrument_idx[current_instrument_name] = 0
                else:
                    current_instrument_idx[current_instrument_name] += 1

                if not self.dict['options'].get('voice_token', False):
                    current_instrument_part = str(current_instrument_idx[current_instrument_name])

            elif key == self.INSTRUMENT_PART:
                # Assuming that instrument part is not used directly in Melody
                current_instrument_part = value

            elif key == self.NOTE_TYPE:
                note_type = value

            elif key == self.NOTE_VAL:
                note_val = int(value)

            elif key == self.NOTE_OCTAVE:
                note_octave = int(value)

            elif key == self.NOTE_AMP:
                note_amp = value

            elif key == self.NOTE_DURATION_NUM:
                note_duration_num = int(value)

            elif key == self.NOTE_DURATION_DEN:
                note_duration_den = int(value)
                current_note_duration = frac(note_duration_num, note_duration_den)
                if current_melody_duration + current_note_duration > chord_duration:
                    delta = current_melody_duration + current_note_duration - chord_duration
                    note_duration_num = int(delta.numerator)
                    note_duration_den = int(delta.denominator)
                    note_duration = current_note_duration - frac(note_duration_num, note_duration_den)
                    if note_duration > 0:
                        note = Note(type=note_type, val=note_val, octave=note_octave, duration=note_duration)
                        note = note.set_amp(note_amp)
                        current_melody.notes.append(note)
                    current_melody_duration += note_duration
                    current_chord.score[current_instrument_name + '__' + current_instrument_part] = current_melody
                else:
                    note = Note(type=note_type, val=note_val, octave=note_octave, duration=current_note_duration)
                    note = note.set_amp(note_amp)
                    current_melody_duration += current_note_duration
                    current_melody.notes.append(note)

                current_instrument = current_instrument_name + '__' + current_instrument_part
                current_chord.score[current_instrument] = current_melody

        # Add the last chord to the score
        if current_chord is not None:
            score.chords.append(current_chord)

        return score