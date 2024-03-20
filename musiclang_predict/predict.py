from musiclang import Score
from musiclang.library import *

from musiclang_predict.chelpers import run_for_n_bars, run_transformer_model, create_transformer, free_transformer_external
import os
import huggingface_hub
from huggingface_hub import hf_hub_download
from musiclang_predict import MusicLangTokenizer

STOP_CHAR = None


TEST_CHORD = (I % I.M)


def get_nb_tokens_chord(tokenizer):
    return len(tokenizer.tokenize_to_bytes(TEST_CHORD))

MIDI_EXTENSIONS = ['mid', 'midi', 'MID', 'MIDI']
XML_EXTENSIONS = ['xml', 'mxl', 'musicxml', 'XML', 'MXL', 'MUSICXML']



class MusicLangPredictor:

    CHORD_CHANGE_CHAR = "_"  # FIXME : It should be generic to the tokenizer
    MELODY_END_CHAR = "0"

    def __init__(self, path, tokenizer_file="tokenizer.bin", model_file="model.bin"):
        self.path = path
        self.tokenizer_path = hf_hub_download(repo_id=self.path, filename=tokenizer_file)
        self.model_path = hf_hub_download(repo_id=self.path, filename=model_file)
        self.pretokenizer = MusicLangTokenizer(self.path)
        self.CHORD_CHANGE_CHAR = self.pretokenizer.tokens_to_bytes('CHORD_CHANGE')
        self.MELODY_END_CHAR = self.pretokenizer.tokens_to_bytes('MELODY_END')

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
        score = self.pretokenizer.tokenize_to_bytes(score, self.pretokenizer) + self.CHORD_CHANGE_CHAR
        return score

    def predict(self, score=None, prompt_chord_range=None, nb_chords=None, nb_tokens: int = 256, temperature=0.9, topp=1.0, rng_seed=0):
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

        transformer_ptr = create_transformer(self.model_path)
        if nb_chords is None:
            score = run_transformer_model(
                transformer_ptr=transformer_ptr,
                attention_already_generated=False,
                tokenizer_path=self.tokenizer_path,
                temperature=temperature,
                topp=topp,
                rng_seed=rng_seed,
                steps=nb_tokens,
                prompt=score,
                post_prompt=None,
                mode="generate",
                system_prompt=None,
                stop_char=None)
        else:
            attention_already_generated = False
            if score is None:
                score = self.CHORD_CHANGE_CHAR
            for i in range(nb_chords):
                score = run_transformer_model(
                    transformer_ptr=transformer_ptr,
                    attention_already_generated=attention_already_generated,
                    tokenizer_path=self.tokenizer_path,
                    temperature=temperature,
                    topp=topp,
                    rng_seed=rng_seed,
                    steps=nb_tokens,
                    prompt=score,
                    post_prompt=None,
                    mode="generate",
                    system_prompt=None,
                    stop_char=self.CHORD_CHANGE_CHAR)
                attention_already_generated = True
        # Untokenize to score
        score = self.pretokenizer.untokenize_from_bytes(score)
        free_transformer_external(transformer_ptr)
        return score

    def ts_to_duration(self, ts):
        from fractions import Fraction
        return Fraction(4 * ts[0], ts[1])

    @property
    def nb_tokens_chord(self):
        return self._nb_tokens_chord

    def chord_to_tokens(self, chord):
        return self.pretokenizer.tokenize_to_bytes(chord, self.pretokenizer)[1:self.nb_tokens_chord]

    def instrument_to_tokens(self, instrument, voice):
        token_instrument = self.pretokenizer.INSTRUMENT_NAME + '__' + instrument
        token_voice = self.pretokenizer.INSTRUMENT_PART + '__' + str(voice)
        return self.pretokenizer.tokens_to_bytes(" ".join([token_instrument]))

    def fix_instrument_name(self, instrument):
        instrument = instrument.lower()
        instrument = instrument.replace(" ", "_")
        if instrument == "drums":
            return "drums_0"
        return instrument


    def midi_to_template(self, midi_file, chord_range=None):
        """
        Convert a midi file to a template usable for the predict_chords_and_instruments method
        :param midi_file:
        :return:
        """
        score = Score.from_midi(midi_file, chord_range=chord_range)
        time_signature = score.config['time_signature'][1], score.config['time_signature'][2]
        chords = score.to_chord_repr().split(' ')
        template = []
        for idx, chord in enumerate(chords):
            instruments = score[idx].instruments
            instruments = [ins.split('__')[0] for ins in instruments]
            template.append((chord, instruments))

        return template, time_signature


    def predict_chords_and_instruments(self, template, time_signature=(4, 4), score=None, prompt_chord_range=None, nb_tokens: int = 4096, temperature=0.9, topp=1.0, rng_seed=0):
        """
        Template is a list of tuple (chords, instrument)
        eg : [("EM", ['piano', 'violin']), ("EM", ['piano', 'violin', 'drums_0'])]

        :param template:
        :param instruments:
        :param time_signature:
        :param score:
        :param prompt_chord_range:
        :param nb_tokens:
        :param temperature:
        :param topp:
        :param rng_seed:
        :return:
        """

        transformer_ptr = create_transformer(self.model_path)
        chord_duration = self.ts_to_duration(time_signature)
        chords = " ".join([chord for chord, instr in template])
        chords = Score.from_chord_repr(chords)
        chords = chords.set_duration(chord_duration)
        chord_tokens = [self.chord_to_tokens(chord) for chord in chords]

        attention_already_generated = False
        if score is not None:
            score = self.parse_score(score, prompt_chord_range)
        else:
            score = self.CHORD_CHANGE_CHAR
        for idx in range(len(template)):
            chord = chord_tokens[idx]
            instruments = template[idx][1]
            idx_instruments = {}
            for idx_inst, inst in enumerate(instruments):
                voice_index = idx_instruments.get(inst, 0)
                idx_instruments[inst] = voice_index + 1
                post_prompt = ""
                if idx_inst == 0:
                    post_prompt = chord
                post_prompt = post_prompt + self.instrument_to_tokens(inst, voice_index)
                generated_text = run_transformer_model(
                    transformer_ptr=transformer_ptr,
                    attention_already_generated=attention_already_generated,
                    tokenizer_path=self.tokenizer_path,
                    temperature=temperature,
                    topp=topp,
                    rng_seed=rng_seed,
                    steps=nb_tokens,
                    prompt=score,
                    post_prompt=post_prompt,
                    mode="generate",
                    system_prompt=None,
                    stop_char=self.MELODY_END_CHAR)

                score = generated_text
                attention_already_generated = True

        score = self.pretokenizer.untokenize_from_bytes(score)
        free_transformer_external(transformer_ptr)

        return score

    def predict_chords(self, chords: str, time_signature=(4, 4), score=None, prompt_chord_range=None, nb_tokens: int = 4096, temperature=0.9, topp=1.0, rng_seed=0):

        transformer_ptr = create_transformer(self.model_path)
        chord_duration = self.ts_to_duration(time_signature)
        chords = Score.from_chord_repr(chords)
        chords = chords.set_duration(chord_duration)
        chord_tokens = [self.chord_to_tokens(chord) for chord in chords]
        attention_already_generated = False
        if score is not None:
            score = self.parse_score(score, prompt_chord_range)
        else:
            score = self.CHORD_CHANGE_CHAR
        for chord in chord_tokens:
            generated_text = run_transformer_model(
                transformer_ptr=transformer_ptr,
                attention_already_generated=attention_already_generated,
                tokenizer_path=self.tokenizer_path,
                temperature=temperature,
                topp=topp,
                rng_seed=rng_seed,
                steps=nb_tokens,
                prompt=score,
                post_prompt=chord,
                mode="generate",
                system_prompt=None,
                stop_char=self.CHORD_CHANGE_CHAR)

            score = generated_text
            attention_already_generated = True

        score = self.pretokenizer.untokenize_from_bytes(score)
        free_transformer_external(transformer_ptr)

        return score

