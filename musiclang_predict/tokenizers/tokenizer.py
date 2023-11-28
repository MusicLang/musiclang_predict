from musiclang import Score, Chord, Note, Melody, Tonality
import os
import tempfile
import json
from fractions import Fraction as  frac
import joblib
import numpy as np
from multiprocessing import Pool
import functools
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import gc


default_options = {
    'chord_change_token': True,
    'melody_end_token': True,
    'chord_duration_token': True,
    'density_token': True,
    'next_chord_token': True,
    'will_end_token': True,
    'dissonance_token': True,
    'amplitude_token': True,
    'average_octave_token': True
}
class MusicLangTokenizer:
    """
    Convert a score into a list of tokens
    """

    CHORD_DEGREE = 'CHORD_DEGREE'
    TONALITY_DEGREE = 'TONALITY_DEGREE'
    TONALITY_MODE = 'TONALITY_MODE'
    CHORD_OCTAVE = 'CHORD_OCTAVE'
    CHORD_DURATION_NUM = 'CHORD_DURATION_NUM'
    CHORD_DURATION_DEN = 'CHORD_DURATION_DEN'

    NEXT_CHORD_DEGREE = 'NEXT_CHORD_DEGREE'
    NEXT_TONALITY_DEGREE = 'NEXT_TONALITY_DEGREE'
    NEXT_TONALITY_MODE = 'NEXT_TONALITY_MODE'
    NEXT_CHORD_OCTAVE = 'NEXT_CHORD_OCTAVE'
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
        if tokenizer_path is not None:
            try:
                with open(tokenizer_path, 'r') as f:
                    self.dict = json.load(f)
            except Exception as e:
                tokenizer_path = hf_hub_download(repo_id=tokenizer_path, filename=hub_tokenizer_path)
                with open(tokenizer_path, 'r') as f:
                    self.dict = json.load(f)

            # Replace str to int for keys of id_to_token
            self.dict['id_to_token'] = {int(k): v for k, v in self.dict['id_to_token'].items()}

        if options is not None:
            self.dict['options'] = options
        elif 'options' not in self.dict:
            self.dict['options'] = default_options

    @property
    def vocab_size(self):
        return len(self.dict.get('token_to_id', {}))

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.dict['token_to_id'][item]
        elif isinstance(item, int):
            return self.dict['id_to_token'][item]
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
        self.dict = {'token_to_id': dict, 'id_to_token': inv_dict, 'options': self.dict['options']}

    def ids_to_tokens(self, ids):
        """
        Convert a list of token ids to a list of tokens
        Parameters
        ----------
        ids

        Returns
        -------

        """
        return [self.dict['id_to_token'][id] for id in ids]

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
        return [self.dict['token_to_id'][token] for token in tokens]



    def train(self, midi_files, output, num_processes=8, resume=True, interceptors=None, **kwargs):
        """
        Train a tokenizer on a list of midi files, and save the tokens ids to files
        Parameters
        ----------
        midi_files

        Returns
        -------
        """
        output_tokens = os.path.join(output, 'tokens')
        output_tokenizer = os.path.join(output, 'tokenizer.json')
        output_ids = os.path.join(output, 'ids')
        # Create directories
        os.makedirs(output_tokens, exist_ok=True)
        os.makedirs(output_ids, exist_ok=True)

        if num_processes > 1:
            if resume:
                # Filter midi files that have already been tokenized
                token_files = [os.path.join(output_tokens, f) for f in os.listdir(output_tokens)]
                token_files = set(token_files)
                midi_files = [midi_file for midi_file in midi_files if os.path.join(output_tokens, os.path.basename(midi_file).replace('.mid', '.txt')) not in token_files]

            self.tokenize_to_files(midi_files, output_tokens, num_processes=num_processes, interceptors=interceptors, **kwargs)
        else:
            self.tokenize_to_files_single(midi_files, output_tokens, **kwargs)
        token_files = [os.path.join(output_tokens, f) for f in os.listdir(output_tokens)]
        self.calculate_tokens_to_ids_dict(token_files)
        # Convert all tokens to ids and save to files
        # Create directory
        os.makedirs(output_ids, exist_ok=True)
        for token_file in tqdm(token_files, 'Saving tokens ids to files'):
            with open(token_file, 'r') as f:
                tokens = f.read().split('\n')
                tokens_ids = self.tokens_to_ids(tokens)
                filename = os.path.basename(token_file)
                # Replace extension to .npy
                filename = filename.replace('.txt', '.npy')
                output_file = os.path.join(output_ids, filename)
                # Save as numpy array
                joblib.dump(np.asarray(tokens_ids, dtype=np.int16), output_file)

        with open(output_tokenizer, 'w') as f:
            json.dump(self.dict, f, indent=4)


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

    def tokenize(self, score):
        if isinstance(score, Chord):
            score = Score(chords=[score])
        tokens = []
        for idx, chord in enumerate(score):
            densities = chord.to_score().extract_densities()
            octave_means = chord.to_score().extract_mean_octaves()
            amplitude_means = chord.to_score().extract_mean_amplitudes()

            tokens_chord = self.tokenize_chord(chord)
            tokens +=  tokens_chord
            if self.dict['options'].get('next_chord_token', False):
                # Last element
                if idx < len(score.chords) - 1:
                    next_chord = score.chords[idx + 1]
                    tokens_next_chord = self.tokenize_next_chord(next_chord)
                    tokens += tokens_next_chord
                else:
                    tokens += [self.WILL_END]

            for ins, melody in chord.score.items():

                ins_name, ins_part = ins.split('__')
                tokens_ins_name = self.INSTRUMENT_NAME + '__' + ins_name
                tokens_ins_part = self.INSTRUMENT_PART + '__' + ins_part
                tokens += [tokens_ins_name, tokens_ins_part]
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
        tokens.append(self.END)

        return tokens

    def tokenize_chord_duration(self, chord_duration):
        token_chord_duration_num = self.CHORD_DURATION_NUM + '__' + str(chord_duration.numerator)
        token_chord_duration_den = self.CHORD_DURATION_DEN + '__' + str(chord_duration.denominator)
        return [token_chord_duration_num, token_chord_duration_den]

    def tokenize_note(self, note):
        note_type = self.NOTE_TYPE + '__' + note.type
        note_degree = self.NOTE_VAL + '__' + str(note.val)
        note_octave = self.NOTE_OCTAVE + '__' + str(note.octave)
        note_amp = self.NOTE_AMP + '__' + note.amp_figure
        note_duration_num = self.NOTE_DURATION_NUM + '__' + str(note.duration.numerator)
        note_duration_den = self.NOTE_DURATION_DEN + '__' + str(note.duration.denominator)
        # Create the list
        tokens = [note_type, note_degree, note_octave, note_amp, note_duration_num, note_duration_den]
        return tokens

    def tokenize_chord(self, chord):
        tokens = []
        if self.dict['options']['chord_change_token']:
            tokens.append(self.CHORD_CHANGE)

        chord_degree = self.CHORD_DEGREE + '__' + str(chord.element)
        chord_octave = self.CHORD_OCTAVE + '__' + str(chord.octave)
        tonality_degree = self.TONALITY_DEGREE + '__' + str(chord.tonality.degree)
        tonality_mode = self.TONALITY_MODE + '__' + chord.tonality.mode

        # Create the list
        tokens += [chord_degree, tonality_degree, tonality_mode, chord_octave]

        if self.dict['options']['chord_duration_token']:
            chord_duration_num = self.CHORD_DURATION_NUM + '__' + str(chord.duration.numerator)
            chord_duration_den = self.CHORD_DURATION_DEN + '__' + str(chord.duration.denominator)
            tokens += [chord_duration_num, chord_duration_den]

        return tokens

    def tokenize_next_chord(self, chord):
        tokens = []

        chord_degree = self.NEXT_CHORD_DEGREE + '__' + str(chord.element)
        chord_octave = self.NEXT_CHORD_OCTAVE + '__' + str(chord.octave)
        tonality_degree = self.NEXT_TONALITY_DEGREE + '__' + str(chord.tonality.degree)
        tonality_mode = self.NEXT_TONALITY_MODE + '__' + chord.tonality.mode

        tokens += [chord_degree, tonality_degree, tonality_mode, chord_octave]

        if self.dict['options']['chord_duration_token']:
            chord_duration_num = self.NEXT_CHORD_DURATION_NUM + '__' + str(chord.duration.numerator)
            chord_duration_den = self.NEXT_CHORD_DURATION_DEN + '__' + str(chord.duration.denominator)
            tokens += [chord_duration_num, chord_duration_den]

        return tokens

    def untokenize(self, tokens):
        score = Score()
        current_chord = None
        current_melody = None
        current_instrument_name = None
        current_instrument_part = None

        for token in tokens:
            # Split token into key and value
            if token in ['END', 'MELODY_END', 'CHORD_CHANGE', 'WILL_END']:
                continue
            key, value = token.split('__')

            if key == self.CHORD_DEGREE:
                if current_chord is not None:
                    score.chords.append(current_chord)
                current_chord = Chord(element=int(value), tonality=Tonality(0, 'M'))

            elif key == self.CHORD_OCTAVE:
                current_chord.octave = int(value)

            elif key == self.TONALITY_DEGREE:
                current_chord.tonality.degree = int(value)  # Assuming Tonality can be constructed from a string

            elif key == self.TONALITY_MODE:
                current_chord.tonality.mode = value

            elif key == self.INSTRUMENT_NAME:
                current_instrument_name = value
                current_melody = Melody(notes=[])

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
                note_duration = frac(note_duration_num, note_duration_den)
                note = Note(type=note_type, val=note_val, octave=note_octave, duration=note_duration)
                note = note.set_amp(note_amp)
                current_melody.notes.append(note)
                current_instrument = current_instrument_name + '__' + current_instrument_part
                current_chord.score[current_instrument] = current_melody

        # Add the last chord to the score
        if current_chord is not None:
            score.chords.append(current_chord)

        return score