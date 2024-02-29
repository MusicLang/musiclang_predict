from musiclang import Score
from musiclang.library import *

from musiclang_predict.chelpers import run_for_n_bars, run_transformer_model
import os
import huggingface_hub
from huggingface_hub import hf_hub_download
from musiclang_predict import MusicLangTokenizer

STOP_CHAR = None
CHORD_CHANGE_CHAR = "_"

TEST_CHORD = (I % I.M)


def get_nb_tokens_chord(tokenizer):
    return len(tokenizer.tokenize_to_bytes(TEST_CHORD))

MIDI_EXTENSIONS = ['mid', 'midi', 'MID', 'MIDI']
XML_EXTENSIONS = ['xml', 'mxl', 'musicxml', 'XML', 'MXL', 'MUSICXML']

class MusicLangPredictor:

    def __init__(self, path, tokenizer_file="tokenizer.bin", model_file="model.bin"):
        self.path = path
        self.tokenizer_path = hf_hub_download(repo_id=self.path, filename=tokenizer_file)
        self.model_path = hf_hub_download(repo_id=self.path, filename=model_file)
        self.pretokenizer = MusicLangTokenizer(self.path)
        self._nb_tokens_chord = get_nb_tokens_chord(self.pretokenizer)

    def parse_score(self, score, prompt_chord_range=None):
        # Tokenize the score to bytes

        if isinstance(score, str) and score.split('.')[-1] in MIDI_EXTENSIONS:
            # Load score
            from musiclang import Score
            score = Score.from_midi(score, chord_range=prompt_chord_range)
        elif isinstance(score, str) and score.split('.')[-1] in XML_EXTENSIONS:
            # Load score
            from musiclang import Score
            if prompt_chord_range is not None:
                raise ValueError("Sorry ... Chord range is not supported yet for XML files, convert it to midi first")
            score = Score.from_xml(score)
        score = self.pretokenizer.tokenize_to_bytes(score, self.pretokenizer) + CHORD_CHANGE_CHAR
        return score

    def predict(self, score=None, prompt_chord_range=None, nb_tokens: int = 256, temperature=0.9, topp=1.0, rng_seed=0):
        """
        Generate a score from a prompt
        :param score: (Optional) MusicLang Score, midi or xml file, default None
        The prompt used to continue the generation on
        :param prompt_chord_range: (Optional) tuple (int, int), default None
        Chord range to use for the prompt
        :param nb_tokens: (Optional) int, default 256
        Number of tokens to generate
        :param temperature: (Optional) float, default 0.9
        Temperature to use for the generation
        :param topp: (Optional) float, default 1.0
        Top-p to use for the generation
        :param rng_seed: (Optional) int, default 0
        Random seed to use for the generation. Use 0 for no seed
        :return: MusicLang Score
        The generated score
        """
        if score is not None:
            score = self.parse_score(score, prompt_chord_range)

        generated_text = run_transformer_model(
            checkpoint_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            temperature=temperature,
            topp=topp,
            rng_seed=rng_seed,
            steps=nb_tokens,
            prompt=score,
            mode="generate",
            system_prompt=None,
            stop_char=STOP_CHAR)

        # Untokenize to score
        score = self.pretokenizer.untokenize_from_bytes(generated_text)
        return score

    def ts_to_duration(self, ts):
        from fractions import Fraction
        return Fraction(4 * ts[0], ts[1])

    @property
    def nb_tokens_chord(self):
        return self._nb_tokens_chord

    def chord_to_tokens(self, chord):
        return self.pretokenizer.tokenize_to_bytes(chord, self.pretokenizer)[1:self.nb_tokens_chord]

    def predict_chords(self, chords: str, time_signature=(4, 4), score=None, prompt_chord_range=None, nb_tokens: int = 4096, temperature=0.9, topp=1.0, rng_seed=0):

        chord_duration = self.ts_to_duration(time_signature)
        chords = Score.from_chord_repr(chords)
        chords = chords.set_duration(chord_duration)
        chord_tokens = [self.chord_to_tokens(chord) for chord in chords]

        if score is not None:
            score = self.parse_score(score, prompt_chord_range)
        else:
            score = CHORD_CHANGE_CHAR

        for chord in chord_tokens:

            score += chord

            generated_text = run_transformer_model(
                checkpoint_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                temperature=temperature,
                topp=topp,
                rng_seed=rng_seed,
                steps=nb_tokens,
                prompt=score,
                mode="generate",
                system_prompt=None,
                stop_char=CHORD_CHANGE_CHAR)

            score = generated_text

        score = self.pretokenizer.untokenize_from_bytes(score)

        return score

