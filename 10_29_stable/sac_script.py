from improc import *
from fake_worm_cont import *

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env = FakeWorm(ep_len=500)
model = SAC(MlpPolicy, env, verbose=2, buffer_size=int(1e4), device='cpu', learning_starts=500,
    tau=5e-3, gamma=0, policy_kwargs={'net_arch':[64,64]}, learning_rate=3e-3, batch_size=128)
eval_r, eval_std = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print('Reward collected was ',eval_r)
model.learn(total_timesteps=int(7000),log_interval=1)
model.save('sac_fakeworm')
eval_r, eval_std = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print('Reward collected was ',eval_r)