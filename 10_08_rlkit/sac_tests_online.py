import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector

from torch import load as load_net
from worm_env_cont import *

def experiment(variant):
    # Sets up experiment with an option to start from a previous run. 
    # Checkpoint in variant is defined before main.

    expl_env = NormalizedBoxEnv(ProcessedWorm(0),obs_mean=0,obs_std=180)
        # Makes observations the networks see range from -1 to 1
    eval_env = expl_env

    if variant['checkpt'] is None:
        obs_dim = expl_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size
        chkpt_log_alpha = None
        M = variant['layer_size']

        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )
    else:
        net = load_net(variant['checkpt'])
        qf1 = net['trainer/qf1']
        qf2 = net['trainer/qf2']
        target_qf1 = net['trainer/target_qf1']
        target_qf2 = net['trainer/target_qf2']
        policy = net['trainer/policy']
        chkpt_log_alpha = net['trainer/fin_log_alpha']

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = MdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        # Takes prev log alpha if it exists
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        log_alpha=chkpt_log_alpha,
        **variant['trainer_kwargs']
    )
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


chkpt = None

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=16,
        replay_buffer_size=int(1E4),
        algorithm_kwargs=dict(
            num_epochs=4,
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=3000,
            num_expl_steps_per_train_loop=1500,
            min_num_steps_before_training=500,
            max_path_length=500,
            batch_size=128,
        ),
        trainer_kwargs=dict(
            discount=0.98,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-3,
            qf_lr=3E-3,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        checkpt=chkpt,
    )
    
    setup_logger('.\\test\\',variant=variant,snapshot_mode='all')
    #ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
