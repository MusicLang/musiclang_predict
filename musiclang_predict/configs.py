# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

CONFIG_MUSICLANG_BASIC = {'out_dir': 'out-musiclang-char', 'eval_interval': 250, 'eval_iters': 200, 'log_interval': 10,
                          'always_save_checkpoint': False, 'wandb_log': False, 'wandb_project': 'musiclang-char',
                          'wandb_run_name': 'mini-gpt', 'dataset': 'musiclang_char', 'batch_size': 64,
                          'block_size': 256,
                          'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'dropout': 0.2, 'learning_rate': 1e-3,
                          'max_iters': 15000, 'lr_decay_iters': 15000, 'min_lr': 1e-4, 'beta2': 0.99,
                          'warmup_iters': 100}

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
