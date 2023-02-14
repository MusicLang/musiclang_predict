from .train_utils import train
from .configs import CONFIG_MUSICLANG_BASIC



class MusicLangBasicTrainer:

    def train(self, **config):

        base_config = dict(CONFIG_MUSICLANG_BASIC)
        base_config.update(**config)
        train(base_config)