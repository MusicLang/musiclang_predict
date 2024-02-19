from transformers import AutoTokenizer



class MusicLangBPETokenizer:

    def __init__(self, tokenizer_path, pretokenizer_path):
        from musiclang_predict import MusicLangTokenizer

        self.pretokenizer = MusicLangTokenizer(pretokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    def tokens_to_bytes(self, tokens: list[str]) -> str:
        """
        Convert a list of tokens to a string of bytes.
        :param tokens:
        :return:
        """
        return self.pretokenizer.tokens_to_bytes(tokens)

    def __call__(self, score, **kwargs) -> list[str]:
        """
        Tokenize the input text.
        :param text: input text.
        :return: list of tokens.
        """
        tokens = self.pretokenizer.tokenize(score)
        bytes_ = self.tokens_to_bytes(tokens)

        return self.tokenizer(bytes_, **kwargs).input_ids

    def ids_to_text(self, ids: list[int]) -> list[str]:
        bytes_ = self.tokenizer.decode(ids).replace(' ', '')
        text = self.pretokenizer.bytes_to_tokens(bytes_)
        return text

    def untokenize(self, ids: list[int]) -> str:
        """
        Convert a list of tokens to a string of bytes.
        :param tokens:
        :return:
        """
        bytes_ = self.tokenizer.decode(ids).replace(' ', '')
        text = self.pretokenizer.bytes_to_tokens(bytes_)
        return self.pretokenizer.untokenize(text)