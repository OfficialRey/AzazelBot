import numpy as np
from rlgym.envs.match import Match

from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, \
    NoTouchTimeoutCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize
from stable_baselines3.ppo import MlpPolicy

from rewards.environmental_rewards import ConstantReward
from rewards.car_rewards import CarSpeedReward, OnGroundReward
from rewards.ball_rewards import VelocityBallToGoalReward, TouchBallReward

# Hyper Parameters
tick_skip = 8
half_life_seconds = 5
fps = int(120 / tick_skip)
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs

# Match Data
team_size = 3
num_agents = team_size * 2
num_instances = 1

# Training Data
target_steps = 1_000_000
steps = target_steps // (num_instances * num_agents)
batch_size = target_steps // 10
training_interval = 25_000_000
save_frequency = 50_000_000
n_epochs = 16

# Global Variables
env: SB3MultipleInstanceEnv
model: PPO
callback: CheckpointCallback
model_name: str = 'RLGymModel'


def init_environment():
    global env
    env = SB3MultipleInstanceEnv(get_match, num_instances)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)


def init_model():
    global model
    global callback
    try:
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="auto",
            custom_objects={
                "n_envs": env.num_envs,
                "n_steps": steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs
            }
        )
        print("Loaded previous model.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 1024, 1024, 512, 512, 512, 512, 256, 256, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=n_epochs,  # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,  # Around this is fairly common for PPO
            ent_coef=0.01,  # From PPO Atari
            vf_coef=1.,  # From PPO Atari
            gamma=gamma,  # Gamma as calculated using half-life
            verbose=3,  # Print out all the info as we're going
            batch_size=batch_size,  # Batch size as high as possible within reason
            n_steps=steps,  # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"  # Uses GPU if available
        )

    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")


def get_match():
    return Match(
        team_size=team_size,
        tick_skip=tick_skip,

        reward_function=CombinedReward(
            (
                # Environmental
                ConstantReward(),

                # Car
                CarSpeedReward(),
                OnGroundReward(),

                # Ball
                VelocityBallToGoalReward(),
                TouchBallReward(),

                # Events
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=5.0,
                    save=30.0,
                    demo=10.0,
                )
            ),
            (
                # Environmental
                0.3,

                # Car
                0.3,
                0.7,

                # Ball
                1.0,
                0.3,

                # Events
                1.0
            )
        ),
        gravity=1,
        game_speed=100,
        obs_builder=AdvancedObs(),
        state_setter=DefaultState(),
        action_parser=DiscreteAction(),
        spawn_opponents=False,
        boost_consumption=1,
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()]
    )


def train_model():
    save_target_count = model.num_timesteps + save_frequency
    while True:
        model.learn(training_interval)
        save_model()

        if model.num_timesteps >= save_target_count:
            model.save(f'models/mmr/{model.num_timesteps}')
            print(f'Saving mmr model. ({model.num_timesteps} steps)')
            save_target_count += save_frequency


def save_model():
    model.save(f'models/{model_name}')
    print(f'Saving model. ({model.num_timesteps} steps)')


if __name__ == '__main__':
    init_environment()
    init_model()
    train_model()
