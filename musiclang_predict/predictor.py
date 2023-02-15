from musiclang import Score
from .sample import ModelLLM


class MusicLangPredictor:
    CHORD_SEP = "+ \n("

    def __init__(self, model):
        self.model = model


    @classmethod
    def load_model(cls, name=None, path=None, update=False, **config):
        model = ModelLLM.load_model(name=name, path=path, update=update, **config)
        return MusicLangPredictor(model)


    def predict(self, score, nb_chords=1, **config):
        """
        Predict The continuation of a score from
        Parameters
        ----------
        score: musiclang.Score or str
              Score on which to do a prediction
        """

        if isinstance(score, str):
            raise NotImplemented('Strings are not supported yet')

        score = score.to_score()
        nb_chords_current = len(score.chords)

        score_str = str(score)
        samples = self.model.sample(start=score_str, num_samples=1, **config)[0]
        samples = samples.replace(';', '+')
        prediction = self.CHORD_SEP.join(samples.split(self.CHORD_SEP)[:-1])

        target_nb_chords = (nb_chords + nb_chords_current)
        new_score = Score.from_str(prediction)
        new_score = new_score[:target_nb_chords]
        new_len_chord = len(new_score.chords)
        if new_len_chord < target_nb_chords:
            return self.predict(new_score, nb_chords=target_nb_chords - new_len_chord, **config)
        else:
            return new_score


class MusicLangPredictorWithTokenizer:
    CHORD_SEP = "+("

    def __init__(self, model):
        self.model = model


    @classmethod
    def load_model(cls, name=None, path=None, update=False, **config):
        model = ModelLLM.load_model(name=name, path=path, update=update, **config)
        return MusicLangPredictor(model)


    def predict(self, score, nb_chords=1, **config):
        """
        Predict The continuation of a score from
        Parameters
        ----------
        score: musiclang.Score or str
              Score on which to do a prediction
        """

        if isinstance(score, str):
            raise NotImplemented('Strings are not supported yet')

        score = score.to_score()
        nb_chords_current = len(score.chords)

        score_str = str(score)
        score_str = score_str.replace(' ', '')
        score_str = score_str.replace('\t', '')
        score_str = score_str.replace('\n', '')
        print(score_str)

        samples = self.model.sample(start=score_str, num_samples=1, **config)[0]

        samples = samples.replace(';', '+')
        samples = samples.replace(' ', '')
        prediction = self.CHORD_SEP.join(samples.split(self.CHORD_SEP)[:-1])

        target_nb_chords = (nb_chords + nb_chords_current)
        new_score = Score.from_str(prediction)
        new_score = new_score[:target_nb_chords]
        new_len_chord = len(new_score.chords)
        if new_len_chord < target_nb_chords:
            return self.predict(new_score, nb_chords=target_nb_chords - new_len_chord, **config)
        else:
            return new_score
