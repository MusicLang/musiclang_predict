from .train_utils import train
from .configs import CONFIG_MUSICLANG_BASIC, CONFIG_MUSICLANG_BASIC_SMALL



class MusicLangBasicTrainer:

    def train(self, out_dir, dataset_dir, **config):

        base_config = dict(CONFIG_MUSICLANG_BASIC)
        base_config.update(**config)
        base_config.update(out_dir=out_dir)
        base_config.update(dataset=dataset_dir)

        train(**base_config)


class MusicLangBasicTrainerXs:

    def train(self, out_dir, dataset_dir, **config):

        base_config = dict(CONFIG_MUSICLANG_BASIC_SMALL)
        base_config.update(**config)
        base_config.update(out_dir=out_dir)
        base_config.update(dataset=dataset_dir)

        train(**base_config)
