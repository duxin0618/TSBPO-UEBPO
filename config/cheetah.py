params = {
    "experiment":{
        "tag": "mb_test",
        "use_gpu": true,
        "seed": null,
        "base_log_dir": "./data/mopo_online_log/"
    },
    "algorithm": {
        "name": "mbpo",
        "class": "MBRLBatchAlgorithm",
        "kwargs": {
            "num_epochs": 300,
            "batch_size": 256,
            "num_eval_steps_per_epoch":8000,
            "num_train_loops_per_epoch": 1000,
            "num_expl_steps_per_train_loop": 1,
            "num_trains_per_train_loop": 20,
            "min_num_steps_before_training": 5000,
            "record_video_freq": 10,
            "silent": false,
            "real_data_ratio": 0.05,
            "train_model_freq": 250,
            "imagine_freq": 250,
            "analyze_freq": 3,
            "train_model_config":{
                "valid_ratio": 0.2,
                "max_valid": 5000,
                "resample": true,
                "batch_size": 256,
                "report_freq": 5,
                "max_not_improve":  20,
                "max_step": 2000,
                "silent": false
            }
        }
    },
    "environment": [
        {
            "name": "expl_env",
            "class": "simple_env",
            "kwargs": {
                "env_name": "mb_ant",
                "reward_scale": 1,
                "known": ["done_function"]
            }
        },
        {
            "name": "eval_env",
            "class": "normalized_vector_env",
            "kwargs": {
                "env_name": "mb_ant",
                "n_env":1,
                "reward_scale": 1
            }
        },
        {
            "name": "video_env",
            "class": "video_env",
            "kwargs": {
                "env_name": "mb_ant"
            }
        }
    ],
    "policy": [
        {
            "name": "init_expl_policy",
            "class": "uniformly_random_policy",
            "kwargs":{
                "env": "$expl_env"
            }
        },
        {
            "name": "policy",
            "class": "gaussian_policy",
            "kwargs": {
                "env": "$expl_env",
                "hidden_layers": [256,256],
                "nonlinearity": "relu"
            }
        },
        {
            "name": "eval_policy",
            "class": "MakeDeterministic",
            "kwargs":{
                "random_policy": "$policy"
            }
        }
    ],
    "value": {
        "name": "qf",
        "class": "EnsembleQValue",
        "kwargs": {
            "env": "$expl_env",
            "hidden_layers": [256,256,256],
            "ensemble_size": 2
        }
    },
    "pool": [
        {
            "name": "pool",
            "class": "ExtraFieldPool",
            "kwargs": {
                "env": "$expl_env",
                "max_size": 1e6,
                "compute_mean_std": true
            }
        },
        {
            "name": "imagined_data_pool",
            "class": "ExtraFieldPool",
            "kwargs": {
                "env": "$expl_env",
                "max_size": 2e6,
                "compute_mean_std": false
            }
        }
    ],
    "collector": [
        {
            "name": "expl_collector",
            "class": "simple_step_collector",
            "kwargs": {
                "env": "$expl_env",
                "policy": "$policy"
            }
        },
        {
            "name": "eval_collector",
            "class": "simple_path_collector",
            "kwargs": {
                "env": "$eval_env",
                "policy": "$eval_policy"
            }
        }
    ],
    "model":{
        "name": "model",
        "class": "PEModel",
        "kwargs": {
            "env": "$expl_env",
            "hidden_layers": [200,200,200,200],
            "nonlinearity": "swish",
            "batch_normalize": false,
            "dense": false
        }
    },
    "models.model_collector":{
        "name": "model_collector",
        "class": "MBPOCollector",
        "kwargs": {
            "model": "$model",
            "policy": "$policy",
            "pool": "$pool",
            "imagined_data_pool": "$imagined_data_pool",
            "depth_schdule": [20,300,1,25],
            "number_sample": 1e5,
            "bathch_size": 5e4
        }
    },
    "trainer":[
        {
            "name": "trainer",
            "class": "ResampleMBPOTrainer",
            "kwargs": {
                "env": "$expl_env",
                "model": "$model",
                "policy": "$policy",
                "qf": "$qf",
                "reward_scale":1,
                "policy_lr": 3e-4,
                "qf_lr": 3e-4,
                "soft_target_tau": 5e-3,
                "use_automatic_entropy_tuning": true,
                "alpha_if_not_automatic": 0,
                "target_entropy": -4
            }
        },
        {
            "name": "model_trainer",
            "class": "PEModelTrainer",
            "kwargs": {
                "env": "$expl_env",
                "model": "$model",
                "lr": 3e-3,
                "weight_decay": [5e-6,1e-5,2e-5,2e-5,3e-5],
                "init_model_train_step": 2e4
            }
        }
    ]
}
