

CONFIG_MUSICLANG_BASIC = {'eval_interval': 250, 'eval_iters': 200, 'log_interval': 10,
                          'always_save_checkpoint': False, 'wandb_log': False, 'wandb_project': 'musiclang-char',
                          'wandb_run_name': 'mini-gpt', 'batch_size': 64,
                          'block_size': 256,
                          'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'dropout': 0.2, 'learning_rate': 1e-3,
                          'max_iters': 10000, 'lr_decay_iters': 10000, 'min_lr': 1e-4, 'beta2': 0.99,
                          'warmup_iters': 100}

CONFIG_MUSICLANG_BASIC_SMALL = {'eval_interval': 250, 'eval_iters': 200, 'log_interval': 10,
                          'always_save_checkpoint': False, 'wandb_log': False, 'wandb_project': 'musiclang-char',
                          'wandb_run_name': 'mini-gpt', 'batch_size': 64,
                          'block_size': 256, 'dtype': 'float16',
                          'n_layer': 3, 'n_head': 2, 'n_embd': 256, 'dropout': 0.2, 'learning_rate': 1e-3,
                          'max_iters': 1000, 'lr_decay_iters': 1000, 'min_lr': 1e-4, 'beta2': 0.99,
                          'warmup_iters': 25}

