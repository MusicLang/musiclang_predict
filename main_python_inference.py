from musiclang_predict import MusicLangPredictor

import time
from musiclang_predict import MusicLangPredictor
from musiclang_predict import corpus
from musiclang_predict.sampler import predict_one_shot

if __name__ == '__main__':
    start = time.time()

    model_path = 'musiclang/musiclang-v2'
    score = predict_one_shot(model_path, device='cpu')

    end = time.time()

    print(f"Generated in {end - start} seconds")

    score.to_midi('test.mid', tempo=110, time_signature=(4, 4))
