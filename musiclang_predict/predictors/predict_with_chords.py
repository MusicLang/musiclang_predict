
from musiclang.library import *
import torch

TEST_CHORD = (I % I.M)


def get_nb_tokens_chord(tokenizer):
    return len(tokenizer(TEST_CHORD))



def predict_with_chords(chord_score, model, tokenizer, chord_duration=4, temperature=1.1, top_k=20, **model_kwargs):


    nb_tokens_chord = get_nb_tokens_chord(tokenizer)
    chord_score = chord_score.to_score().set_duration(chord_duration)

    stop_token = tokenizer.tokenizer(tokenizer.tokens_to_bytes(['CHORD_CHANGE']))['input_ids']
    # We generate one chord per one chord
    eos_token_id = int(stop_token[0])
    result = []

    for chord in chord_score:
        # We generate the chord
        chord_tokens = tokenizer(chord.to_score())[:nb_tokens_chord]
        result += chord_tokens
        # result += [67]

        new_ids = model.generate(torch.tensor([result]), do_sample=True,
                                 eos_token_id=eos_token_id,
                                 temperature=temperature, top_k=top_k,
                                 max_new_tokens=500,
                                 **model_kwargs)[0].tolist()[:-1]

        result = new_ids

    score = tokenizer.ids_to_score(result)
    return score

