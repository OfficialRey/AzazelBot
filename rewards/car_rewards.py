import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import CEILING_Z, CAR_MAX_SPEED
from rlgym_compat import GameState, PlayerData


class OnGroundReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            return 1


class CarSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        print(player.car_data.linear_velocity / CAR_MAX_SPEED)
        return player.car_data.linear_velocity / CAR_MAX_SPEED
