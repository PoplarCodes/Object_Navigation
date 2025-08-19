scenes = {}
scenes["train"] = [
    'Allensville',
    'Beechwood',
    'Benevolence',
    'Coffeen',
    'Cosmos',
    'Forkland',
    'Klickitat',
    'Lakeville',
    'Merom',
    'Mifflinburg',
    'Newfields',
    'Onaga',
    'Pinesdale',
    'Pomaria',
    'Ranchester',
    'Shelbyville',
    'Stockman',
    'Woodbine',
]

scenes["val"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

# 物体与可能房间的先验映射，用于根据目标物体推测其可能存在的房间
object_room_map = {
    "chair": ["living room", "dining room", "study", "bedroom"],
    "couch": ["living room"],
    "potted plant": ["living room", "office", "balcony"],
    "bed": ["bedroom"],
    "toilet": ["bathroom"],
    "tv": ["living room", "bedroom"],
    "dining-table": ["dining room", "kitchen"],
    "oven": ["kitchen"],
    "sink": ["kitchen", "bathroom"],
    "refrigerator": ["kitchen"],
    "book": ["study", "living room"],
    "clock": ["living room", "bedroom", "kitchen"],
    "vase": ["living room", "dining room"],
    "cup": ["kitchen", "dining room"],
    "bottle": ["kitchen", "dining room"],
}
# 使用方式：通过 object_room_map["bed"] 获取 ["bedroom"] 等房间列表

# 房间名称到语义图通道索引的映射，便于将房间先验投射到语义图上
room_channel_map = {
    "living room": 0,
    "dining room": 1,
    "study": 2,
    "bedroom": 3,
    "bathroom": 4,
    "kitchen": 5,
    "office": 6,
    "balcony": 7,
}

# 物体和房间类别数量，便于统一引用
# 物体类别总数
NUM_OBJECT_CATEGORIES = len(coco_categories)
# 房间类别总数
NUM_ROOM_CATEGORIES = len(room_channel_map)

# 需要限制最大掩码面积的小物体类别索引
small_object_indices = {
    coco_categories["chair"],
    coco_categories["potted plant"],
}


color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]
