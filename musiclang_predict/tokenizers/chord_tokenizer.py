import re
import itertools


CHORD_REGEX = re.compile(r'\([^)]+%[^)]+\)')


class ChordTokenizer:
    """Chord tokenizer transforms a score in text format to a list of tokens"""

    def tokenize(self, text):
        """Tokenize a text (or score) to extract chord tokens

        Parameters
        ----------
        text :
            Score or str, if Score first convert it to its string representation

        Returns
        -------
        tokens: List[str]
                List of tokens

        """
        chords = re.findall(CHORD_REGEX, str(text))
        # Deduplicate the chords, because we don't care about rythm here
        deduplicated_chords = [k for k, _g in itertools.groupby(chords)]
        return deduplicated_chords


    def tokenize_all(self, texts):
        """Tokenize all texts

        Parameters
        ----------
        texts :
            return: List of List of tokens

        Returns
        -------
        tokens_list: List[List[str]]
            A list of token per text

        """
        data = []
        for text in texts:
            data.append(self.tokenize(text))

        return data

    def tokenize_file(self, file):
        """Tokenize one file

        Parameters
        ----------
        file : str
            Filepath where to get the tokens

        Returns
        -------
        tokens: List[str]
                List of tokens

        """
        with open(file, 'r') as f:
            return self.tokenize(f.read())

    def tokenize_directory(self, directory):
        """
        Tokenize a directory, call the tokenize_file method for each text in the directory.

        Parameters
        ----------
        directory : str
            Directory to tokenize
            

        Returns
        -------

        data: List[List[tokens]]
             A list of token per text

        """
        import os
        files = [os.path.join(dirpath, file)
                 for (dirpath, dirnames, filenames) in os.walk(directory)
         for file in filenames]

        data = []
        for file in files:
            data.append(self.tokenize_file(file))

        return data


class ChordDetokenizer:
    """Convert tokens from chord tokenizer to chords"""

    def detokenize(self, tokens):
        """
        Convert a tokens list to a musiclang score.

        Parameters
        ----------
        tokens :
            

        Returns
        -------
        score: musiclang.Score
               Score detokenized (only chords)

        """
        from musiclang.write.library import I, II, III, IV, V, VI, VII
        return sum(eval('+'.join(tokens)), None)