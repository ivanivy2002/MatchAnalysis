import sys

sys.path.append("../")
from utils import get_center, measure_distance, distance_feet_point


class PlayerBallAssigner:
    def __init__(self):
        self.max_p_b_dist = 69

    def assign_ball_2_player(self, players, ball):
        ball = get_center(ball)
        dist_list = []
        for id, player in players.items():
            player_bbox = player["bbox"]
            dist = distance_feet_point(player_bbox, ball)
            dist_list.append((id, dist))
        min_id, min_dist = min(dist_list, key=lambda x: x[1])
        if 0 <= min_dist < self.max_p_b_dist:
            return min_id
