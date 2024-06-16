import cv2
from sklearn.cluster import KMeans


class TeamAssigner:
    # 详见 analysis/color_assign.ipynb
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_cluster(self, image):   # 获取聚类
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, n_init=1, init="k-means++").fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):    # 获取球员颜色
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half = image[0 : int(image.shape[0] / 2), :, :]
        # 聚类
        kmeans = self.get_cluster(top_half)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_cluster = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team(self, frame, player_detection):   # 分成两队，给队伍分配颜色
        player_color_list = []
        for _, player_detection in player_detection.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_color_list.append(player_color)

        # 分成两队
        kmeans = KMeans(n_clusters=2, n_init=1, init="k-means++")
        kmeans.fit(player_color_list)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):   # 获取球员队伍
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id
        return team_id
