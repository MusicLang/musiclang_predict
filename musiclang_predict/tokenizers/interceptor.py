
def piano_interceptor(score):
    """
    Before tokenization, transform all tracks into piano tracks except drums

    Parameters
    ----------
    score

    Returns
    -------

    """

    score = score[score.instruments]

    drums = score.get_instrument_names(['drums_0'])
    score = score.remove_drums()

    pitch_statistics = score.get_pitch_statistics()
    instruments = sorted(pitch_statistics, key=lambda x: pitch_statistics[x][2])

    instrument_replace_dict = {}
    instrument_idx = {}
    instruments_selection = []
    idx = -1
    for instrument in instruments:
        if instrument not in instrument_idx:
            idx += 1
            instrument_replace_dict[instrument] = f'piano__{idx}'
            instruments_selection.append(f'piano__{idx}')

    # Replace instruments in score
    score_instrument = score.replace_instruments(**instrument_replace_dict)
    score_instrument = score_instrument[instruments_selection]
    if len(drums.instruments) > 0:
        score_instrument = score_instrument.project_on_score(drums, voice_leading=False, keep_score=True)

    score_instrument = score_instrument.remove_silenced_instruments()

    return score_instrument