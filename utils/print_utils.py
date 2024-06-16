def print_structure(d, indent=0, max_items=5):
    if isinstance(d, dict):
        keys = list(d.keys())
        if len(keys) > max_items:
            keys = keys[:max_items] + ['...']
        for key in keys:
            print('  ' * indent + str(key))
            if key == '...':
                break
            value = d[key]
            if isinstance(value, (dict, list)):
                print_structure(value, indent + 1, max_items)
    elif isinstance(d, list):
        if len(d) > max_items:
            indices = list(range(max_items)) + ['...']
        else:
            indices = list(range(len(d)))
        for index in indices:
            print('  ' * indent + (f'[{index}]' if index != '...' else '...'))
            if index == '...':
                break
            value = d[index]
            if isinstance(value, (dict, list)):
                print_structure(value, indent + 1, max_items)
    else:
        print('  ' * indent + str(d))

# # 使用示例
# tracks = {
#     "players": [
#         {1: {"bbox": [100, 200, 300, 400], "has_ball": True}},
#         {2: {"bbox": [150, 250, 350, 450], "has_ball": False}}
#     ] + [{} for _ in range(18)],
#     "referees": [{} for _ in range(20)],
#     "ball": [{} for _ in range(20)]
# }

# print_structure(tracks)
