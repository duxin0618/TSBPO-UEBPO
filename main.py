import argparse
import importlib
import runner

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()

module = importlib.import_module(args.config)
params = getattr(module, 'params')
universe, domain, task = params['universe'], params['domain'], params['task']

NUM_EPOCHS_PER_DOMAIN = {
    'Hopper': 300,
    'Walker2d': 300,
    'Ant': 300,
    'Swimmer': 300,
    'InvertedPendulum': 300,
    'HalfCheetah': 300,
    'Humanoid': int(1e3),
}

NUM_INITIAL_EXPLORATION_STEPS = {
    'Hopper': 5000,
    'Walker2d': 5000,
    'Ant': 5000,
    'Swimmer': 2000,
    'InvertedPendulum': 300,
    'HalfCheetah': 5000,
    'Humanoid': 5000,
}

Q_FUNCTION_HIDDEN_SIZE = {
    'Hopper': (256, 256),
    'Walker2d': (256, 256),
    'Ant': (256, 256),
    'Swimmer': (256, 256),
    'InvertedPendulum': (256, 256),
    'HalfCheetah': (256, 256),
    'Humanoid': (256, 256, 256),
}

MODEL_HIDDEN_DIM = {
    'Hopper': 200,
    'Walker2d': 200,
    'Ant': 200,
    'Swimmer': 200,
    'InvertedPendulum': 200,
    'HalfCheetah': 200,
    'Humanoid': 400,
}

params['kwargs']['n_epochs'] = NUM_EPOCHS_PER_DOMAIN[domain]
params['kwargs']['n_initial_exploration_steps'] = NUM_INITIAL_EXPLORATION_STEPS[domain]
params['kwargs']['reparameterize'] = True
params['kwargs']['lr'] = 3e-4
params['kwargs']['target_update_interval'] = 1
params['kwargs']['tau'] = 5e-3
params['kwargs']['store_extra_policy_info'] = False
params['kwargs']['action_prior'] = 'uniform'


variant_spec = {
        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
        },
        'policy_params': {
            'type': 'GaussianPolicy',
            'kwargs': {
                'hidden_layer_sizes': (256, 256),
                'squash': True,
            }
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': Q_FUNCTION_HIDDEN_SIZE[domain],
            }
        },
        'algorithm_params': params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': 1000,
                'min_pool_size': 300,
                'batch_size': 256,
            }
        },
        'run_params': {
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN[domain] // 10,
            'checkpoint_replay_pool': False,
        },
        'dir_path': {
            'log_dir_test': './log/testcode_scqd_3/',
            'log_dir_mbpo': './log/mbpo/',
            'log_dir_ablation': './log/ablation/',
            'log_dir': './log/',
            'store_dir': './log/samples/'
        },
        'hidden_dim': MODEL_HIDDEN_DIM[domain]
    }

exp_runner = runner.ExperimentRunner(variant_spec)

diagnostics = exp_runner.train()
