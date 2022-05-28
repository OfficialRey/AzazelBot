import numpy as np
from rlgym.utils.reward_functions.common_rewards import RewardFunction
from rlgym_compat import GameState, PlayerData


class ConstantReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 1
