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

class MusicLangPredictor:

    def __init__(self, path, tokenizer_file="tokenizer.bin", model_file="model.bin"):
        self.path = path
        self.tokenizer_path = hf_hub_download(repo_id=self.path, filename=tokenizer_file)
        self.model_path = hf_hub_download(repo_id=self.path, filename=model_file)
        self.pretokenizer = MusicLangTokenizer(self.path)
        self._nb_tokens_chord = get_nb_tokens_chord(self.pretokenizer)

    def predict(self, score=None, nb_tokens: int = 256, temperature=0.95, topp=1.0, rng_seed=0):

        if score is not None:
            # Tokenize the score to bytes
            score = self.pretokenizer.tokenize_to_bytes(score, self.pretokenizer)

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
        print(generated_text)
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

    def predict_chords(self, chords: str, time_signature=(4, 4), score=None, nb_tokens: int = 4096, temperature=0.95, topp=1.0, rng_seed=0):

        chord_duration = self.ts_to_duration(time_signature)
        chords = Score.from_chord_repr(chords)
        chords = chords.set_duration(chord_duration)
        chord_tokens = [self.chord_to_tokens(chord) for chord in chords]

        if score is not None:
            score = self.pretokenizer.tokenize_to_bytes(score, self.pretokenizer)
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

