def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_width(bbox):
    x1, _, x2, _ = bbox
    return x2 - x1


import cv2


def avg_digit_fontsize(
    font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1
):  # 字体, 缩放比例, 粗细
    # 数字列表
    digits = [str(i) for i in range(10)]
    # 计算每个数字的宽度和高度
    widths = []
    heights = []
    for digit in digits:
        (width, height), baseLine = cv2.getTextSize(digit, font, fontScale, thickness)
        widths.append(width)
        heights.append(height)
    # 计算平均宽度和高度
    average_width = sum(widths) / len(widths)
    average_height = sum(heights) / len(heights)
    return average_width, average_height



def measure_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def distance_feet_point(bbox, point):
    dist1 = measure_distance(point, (bbox[0], bbox[-1]))
    dist2 = measure_distance(point, (bbox[2], bbox[-1]))
    dist = min(dist1, dist2)
    return dist