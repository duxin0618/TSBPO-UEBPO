params = {
    'type': 'SCQDMBPO',
    'universe': 'gym',
    'domain': 'HalfCheetah',
    'task': 'v2',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 40,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -3,
        'max_model_t': None,

        'rollout_schedule': [1, 30, 1, 3],

        'po_stop_epoch': 200,

        'use_src': True,
        'src_iter_schedule': [1, 10, 1, 1],
        'src_fix_iter': True,
        'src_max_length': 10,
        'src_min_length': 2,
        'src_pool_size': 50e3,
        'src_min_fake_sample_size': 20e3,
        'scheme': 1,  # 0 is optimistic, 1 is conservative
        'use_reward_weight': True,
        'reward_weight': 0.999,
        'plan': 7,
        'src_epoch_stop': 30,
        'Q_tau_info': [1, 30, 0.15, 0.15],
        'ratios_threshold': 1.3,
        'src_reset_rollout_length': -1  # 0 is false,
    }
}
