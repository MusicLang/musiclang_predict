from musiclang import Score

def midi_file_to_template(midi_file, chord_range=None, max_instruments=8, quantization=(4, 3)):
    """
    Extract a song template from a midi file. It will extract the chord progression, the orchestration,
    The average density, the average amplitude, the average octave for each instrument of each bar.

    It will also extract metadata about the soundtrack like the tonality, the tempo and the time signature.
    :param midi_file: str, path to midi file
    :param chord_range: tuple, range of chords to extract (start, end) (default=None)
    :param max_instruments: int, maximum number of instruments to extract (default=8)
    :return: dict, template
    """
    score_prompt = Score.from_midi(midi_file, quantization=quantization, chord_range=chord_range)
    return score_to_template(score_prompt, max_instruments=max_instruments)


def score_to_template(score, max_instruments=8):
    """
    Extract a song template from a musiclang score. It will extract the chord progression, the orchestration,
    The average density, the average amplitude, the average octave for each instrument of each bar.

    It will also extract metadata about the soundtrack like the tonality, the tempo and the time signature.
    :param score: str, path to midi file
    :param chord_range: tuple, range of chords to extract (start, end) (default=None)
    :param max_instruments: int, maximum number of instruments to extract (default=8)
    :return: dict, template
    """
    score_prompt = score.to_score()
    densities_per_chords = [chord.to_score().extract_densities() for chord in score_prompt]
    amplitudes_per_chords = [chord.to_score().extract_mean_amplitudes() for chord in score_prompt]
    octaves_per_chords = [chord.to_score().extract_mean_octaves() for chord in score_prompt]
    tonality, chord_list = score_prompt.to_romantext_chord_list()
    data_chords = [
        {
            "orchestration": [{'instrument_name': key.split('__')[0],
                               'instrument_voice': int(key.split('__')[1]),
                               'amplitude': amplitudes_per_chords[idx][key],
                               'octave': octaves_per_chords[idx][key],
                               'density': float(val)}
                              for key, val in densities_per_chord.items()],
            "chord": chord_list[idx]
        }
        for idx, densities_per_chord in enumerate(densities_per_chords)

    ]

    for data_chord in data_chords:
        data_chord['orchestration'] = data_chord['orchestration'][:max_instruments]

    ts = score_prompt.config['time_signature']
    if len(ts) == 4:
        ts = ts[1], ts[2]
    data = {
        'tonality': tonality,
        'tempo': score_prompt.config['tempo'],
        'time_signature': ts,
        'chords': data_chords
    }
    return data
