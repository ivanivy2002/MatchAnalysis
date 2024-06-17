import sys

sys.path.append("../")
from utils import get_center, measure_distance, distance_feet_point


class PlayerBallAssigner:
    def __init__(self):
        self.max_p_b_dist = 69

    def assign_ball_2_player(self, players, ball):
        ball = get_center(ball)
        min_dist = 999999
        for id, player in players.items():
            player_bbox = player["bbox"]
            dist = distance_feet_point(player_bbox, ball)
            if dist > self.max_p_b_dist:
                continue
            if dist < min_dist:
                min_dist = dist
                min_id = id
        if min_dist < self.max_p_b_dist:
            return min_id
        else:
            return None
