

from musiclang import Score
from transformers import StoppingCriteria, StoppingCriteriaList
import time
from musiclang import Chord
from fractions import Fraction as frac

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids, scores, **kwargs):
        stop_count = 0
        for stop in self.stops:
            stop_count += (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False


def extract_max_density_instruments(score):
    density = score.extract_densities()
    max_density_instrument_name = max(density, key=density.get).split('__')[0]
    fixed_parts = [ins for ins in score.instruments if ins.split('__')[0] ==max_density_instrument_name]
    return fixed_parts

def predict_melody(score_prompt, instrument, model, tokenizer, prompt_size=1024, context_size=2048, temperature=0.9, **kwargs):
    chord_duration = score_prompt.chords[0].duration
    inputs = tokenizer.tokenize_to_ids(score_prompt)[:-1]
    score, inputs = predict_melody_from_inputs(inputs, chord_duration, instrument,
                                               model, tokenizer, prompt_size=prompt_size,
                                                context_size=context_size, temperature=temperature, **kwargs
                                               )
    return score


def predict_melody_from_inputs(inputs, chord_duration, instrument, model, tokenizer, density=None,
                               octave=None, amplitude=None,
                               prompt_size=1024, context_size=2048, temperature=0.9, **kwargs):
    # Add density token
    if density is not None and tokenizer.dict['options'].get('density_token', False):
        if isinstance(density, str):
            density_str = density
        else:
            density_str = tokenizer.density_to_density_str(density)
        density_token = tokenizer['DENSITY__' + str(density_str)]
        inputs += [density_token]
    if octave is not None and tokenizer.dict['options'].get('average_octave_token', False):
        octave_token = tokenizer['AVERAGE_OCTAVE__' + str(octave)]
        inputs += [octave_token]
    if amplitude is not None and tokenizer.dict['options'].get('average_amplitude_token', False):
        amplitude_token = tokenizer['AVERAGE_AMPLITUDE__' + str(amplitude)]
        inputs += [amplitude_token]

    nb = context_size - prompt_size
    end_token = tokenizer['END']
    generation_config = dict(
        max_new_tokens=nb,  # extends samples by nb tokens
        min_new_tokens=0,
        do_sample=True,  # but sample instead
        temperature=temperature,
        pad_token_id=end_token,
    )

    if tokenizer.dict['options'].get('melody_end_token', False):
        stop_words_ids = [tokenizer['MELODY_END']]
    else:
        stop_words_ids_instruments = [val for i, val in tokenizer.dict['token_to_id'].items() if
                                      i.startswith('INSTRUMENT_NAME__') or i.startswith('INSTRUMENT_PART__')]
        stop_words_ids_chords = [val for i, val in tokenizer.dict['token_to_id'].items() if i.startswith('CHORD_DEGREE__')]
        stop_words_ids = stop_words_ids_instruments + stop_words_ids_chords

    nb = context_size - prompt_size
    instrument_name, part = instrument.split('__')
    instrument_id = tokenizer['INSTRUMENT_NAME__' + instrument_name]
    instrument_part = tokenizer['INSTRUMENT_PART__' + str(part)]

    inputs += [instrument_id, instrument_part]

    # Load model
    score, inputs = model_pred_one_shot(inputs, nb, stop_words_ids, model,
                        tokenizer, chord_duration, generation_config, nb_stop_word=1, **kwargs)
    return score, inputs

def predict_with_fixed_instruments_no_prompt(instruments, chord_duration, model, tokenizer, prompt_size=1024, context_size=2048, temperature=0.9, n_chords=1,
                                   chords=None,
                                   **kwargs):

    inputs = []
    score = None
    for idx, chord in enumerate(chords):
        # FIXME : New chord not possible, to fix with next tokenizer
        inputs += tokenizer.tokens_to_ids(tokenizer.tokenize_chord(chord.set_duration(chord_duration)))
        if tokenizer.dict['options'].get('next_chord_token', False) and idx < len(chords) - 1:
            inputs += tokenizer.tokens_to_ids(tokenizer.tokenize_next_chord(chords[idx + 1].set_duration(chord_duration)))

        for instrument in instruments:
            score, inputs = predict_melody_from_inputs(inputs, chord_duration, instrument, model, tokenizer,
                                                       prompt_size=prompt_size,
                                                       context_size=context_size, temperature=temperature, **kwargs)
    return score

def predict_with_template(template,
                          model, tokenizer,
                          prompt=None,
                          prompt_included_in_template=True,
                          prompt_size=1024, context_size=2048, temperature=0.9
                          ):
    """
    Predict with a template of instruments and chord value for each chord of the score.
    Allow the music to follow a template.
    Parameters
    ----------
    template: dict
    chords: list of Chord
    model: transformers.GPT2LMHeadModel
    tokenizer: MusicLangTokenizer
    prompt: MusicLang.Score
    prompt_included_in_template: bool
    prompt_size: int
    context_size: int
    temperature: float

    Returns
    -------
    score: MusicLang.Score

    """

    ts_num, ts_den = template['time_signature']
    time_signature = template['time_signature']
    chord_duration = frac(4 * ts_num, ts_den)
    tonality = template['tonality']
    inputs = []

    # If prompt add it to inputs
    if prompt is not None:
        inputs += tokenizer.tokenize_to_ids(prompt)[:-1]
        if prompt_included_in_template:
            template['chords'] = template['chords'][len(prompt.chords):]

    score = None
    for idx, bar in enumerate(template['chords']):
        chord = Chord.from_romantext(bar['chord'], tonality=tonality, time_signature=time_signature)
        instruments_for_chord = bar['orchestration']
        inputs += tokenizer.tokens_to_ids(tokenizer.tokenize_chord(chord.set_duration(chord_duration)))
        if tokenizer.dict['options'].get('next_chord_token', False) and idx < len(template['chords']) - 1:
            next_chord = Chord.from_romantext(template['chords'][idx + 1]['chord'],
                                              tonality=tonality, time_signature=time_signature)
            inputs += tokenizer.tokens_to_ids(
                tokenizer.tokenize_next_chord(next_chord.to_chord().set_duration(chord_duration)))
        for instrument in instruments_for_chord:
            density = instrument['density']
            octave = instrument['octave']
            amplitude = instrument['amplitude']
            instrument = instrument['instrument_name'] + '__' + str(instrument['instrument_voice'])
            score, inputs = predict_melody_from_inputs(inputs, chord_duration, instrument, model, tokenizer,
                                                       density=density,
                                                       octave=octave,
                                                       amplitude=amplitude,
                                                       prompt_size=prompt_size,
                                                       context_size=context_size, temperature=temperature)
    return score


def predict_chords(model, tokenizer, nb_chords=4, prompt=None, temperature=0.9):
    from musiclang_predict import MusicLangTokenizer
    msl_tokenizer = MusicLangTokenizer()

    if prompt is None:
        nb_chord_init = 0
        prompt = "CHORD_CHANGE"
    else:
        nb_chord_init = len(prompt.chords)
        prompt = " ".join(msl_tokenizer.tokenize_chords(prompt))

    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    nb_tokens = (nb_chords + 1 + nb_chord_init) * 6
    result = model.generate(ids, do_sample=True, temperature=temperature,
                            max_new_tokens=nb_tokens,
                            )[0]
    seq = tokenizer.decode(result)
    score = msl_tokenizer.untokenize(prompt + ' ' + seq)
    score = score[:nb_chords]

    return score


def predict_with_fixed_instruments(score_prompt, model, tokenizer, prompt_size=1024, context_size=2048, temperature=0.9, n_chords=1,
                                   chords=None,
                                   **kwargs):
    instruments = score_prompt[-1:].instruments  # Use last instruments by default
    chord_duration = score_prompt.chords[0].duration
    score = score_prompt

    inputs = tokenizer.tokenize_to_ids(score)[:-1]
    if chords is None:
        chords = [score_prompt[-1].to_chord() for i in range(n_chords)]

    for chord in chords:
        # FIXME : New chord not possible, to fix with next tokenizer
        inputs += tokenizer.tokenize_to_ids(chord.set_duration(chord_duration))[:6]
        for instrument in instruments:
            score, inputs = predict_melody_from_inputs(inputs, chord_duration, instrument, model, tokenizer, prompt_size=prompt_size,
                                   context_size=context_size, temperature=temperature, **kwargs)
        # Add chord tokens
    return score


def model_pred_one_shot(inputs, nb, stop_words_ids, model, tokenizer, chord_duration, generation_config,
                        nb_stop_word=1,
                        **kwargs):
    import torch

    from musiclang.transform import create_counterpoint_on_score
    prompt = inputs[-nb:]
    remaining = len(inputs) - len(prompt)
    inputs = inputs[:remaining]
    # Load model
    prompt = torch.tensor([prompt], dtype=torch.long)
    nb_encounters = len([i for i in prompt[0] if i in stop_words_ids])
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids, encounters=nb_encounters + nb_stop_word)])

    tokens = model.generate(prompt, stopping_criteria=stopping_criteria, **generation_config)[0]  # (N,T)
    new_ids = tokens.tolist()
    inputs += new_ids

    # Correct inputs
    score = tokenizer.ids_to_score(inputs[:-1])  # Remove stop token
    score = score.arrange_chords_duration(chord_duration)
    drums = score.get_instrument_names(['drums_0'])
    score_without_drums = score.normalize_instruments().remove_drums()


    if len(score_without_drums.instruments) > 0:
        score = score_without_drums
        fixed_parts = extract_max_density_instruments(score)
        #score = create_counterpoint_on_score(score, fixed_parts=fixed_parts)
        # Reproject the drums
        if len(drums.instruments) > 0:
            score = score.project_on_score(drums, voice_leading=False, keep_score=True)
    score = score.remove_silenced_instruments()
    inputs = tokenizer.tokenize_to_ids(score, include_end=False)

    return score, inputs


def predict(model, tokenizer, prompt=None,
            chord_duration=4,
            prompt_size=1024, context_size=2048, nb_chords=1, temperature=0.9, **kwargs):
    """
    Predict continuation of a score

    Parameters
    ----------
    The score to continue
    model: transformers.GPT2LMHeadModel
    The model to use for prediction
    tokenizer: MusicLangTokenizer
    The tokenizer to use for prediction
    prompt: musiclang.Score
    The score to continue
    chord_duration: int
    The duration of the chords to predict in number of quarter notes
    prompt_size: int
    The size of the prompt to use for prediction
    context_size: int
    The size of the context to use for prediction
    nb_chords: int
    The number of chords to predict
    temperature: float
    The temperature to use for prediction
    kwargs

    Returns
    -------

    """
    stop_words_ids = [tokenizer['END'], tokenizer['CHORD_CHANGE']]
    nb = context_size - prompt_size
    generation_config = dict(
        max_new_tokens=nb,  # extends samples by nb tokens
        min_new_tokens=0,
        do_sample=True,  # but sample instead
        temperature=temperature,
        pad_token_id=tokenizer['END'],
    )

    if prompt:
        chord_duration = prompt.chords[0].duration
        inputs = tokenizer.tokenize_to_ids(prompt, include_end=False)
    else:
        inputs = []
    score = None
    for i in range(nb_chords):
        inputs = inputs + [tokenizer['CHORD_CHANGE']]
        score, inputs = model_pred_one_shot(inputs, nb, stop_words_ids, model,
                                            tokenizer, chord_duration, generation_config, nb_stop_word=1, **kwargs)

    return score
