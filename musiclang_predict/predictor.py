from musiclang import Score
from .sample import ModelLLM


class MusicLangPredictor:
    CHORD_SEP = ")+ \n("

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
            score = Score.from_str(score).to_score()  # Normalize the score

        score = score.to_score()
        if len(score.chords) > 0:
            last_chord_duration = len(str(score[-1]))
        else:
            last_chord_duration = 100

        score_str = str(score)
        samples = self.model.sample(self, start=score, max_new_tokens=last_chord_duration, num_samples=1, **config)[0]

        prediction = self.CHORD_SEP.join(samples.split(self.CHORD_SEP)[:-1]) + self.CHORD_SEP

        return prediction
        # Find chords
        # Remove last chord
