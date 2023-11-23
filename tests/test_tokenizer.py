from musiclang_predict import MusicLangTokenizer
from musiclang.library import *

def test_tokenizer():
    tokenizer = MusicLangTokenizer()
    score = (I % I.M).o(-1)(
        piano__0=s0.o(1).e + s1.e + s2,
        violin__1=s3.o(1).e + h3.e.p + s5,
        piano__5=s6.o(1).e + s3.e + s2,
    ) * 2

    tokens = tokenizer.tokenize(score)
    score_parsed = tokenizer.untokenize(tokens)

    assert score == score_parsed