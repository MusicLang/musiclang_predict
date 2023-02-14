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

        prediction = self.CHORD_SEP.join(samples.split(self.CHORD_SEP)[:-1])

        new_score = Score.from_str(prediction)
        new_score = new_score[:nb_chords + nb_chords_current]
        if len(new_score.chords) < nb_chords + nb_chords_current:
            return self.predict(new_score, nb_chords=nb_chords - nb_chords_current, **config)

