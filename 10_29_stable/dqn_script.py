from improc import *
from fake_worm import *

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env = FakeWorm(ep_len=500)
model = DQN('MlpPolicy', env, verbose=1, buffer_size=int(1e4), learning_rate=3e-3, learning_starts=500, batch_size=128,
    gamma=0, train_freq=1, target_update_interval=1, device='cpu', policy_kwargs={'net_arch':[64,64]},
    tau=5e-3)
eval_r, eval_std = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print('Reward collected was ',eval_r)
model.learn(total_timesteps=int(8000),log_interval=1)
model.save('dqn_fakeworm')
eval_r, eval_std = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print('Reward collected was ',eval_r)
