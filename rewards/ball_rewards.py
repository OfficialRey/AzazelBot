import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BLUE_GOAL_BACK, ORANGE_GOAL_BACK, BALL_MAX_SPEED, BACK_NET_Y, BACK_WALL_Y, \
    BALL_RADIUS, BLUE_TEAM, ORANGE_TEAM, CEILING_Z
from rlgym_compat import GameState, PlayerData


class VelocityBallToGoalReward(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))


class TouchBallReward(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return (np.linalg.norm(state.ball.linear_velocity) / BALL_MAX_SPEED * (
                    state.ball.position[2] / (CEILING_Z * 2))) if player.ball_touched else 0


class DistanceBallToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return np.exp(-0.5 * dist / BALL_MAX_SPEED)
