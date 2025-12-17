import json
import random
from collections import defaultdict

with open("./data_split/train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
with open("./data_split/test_data.json", "r", encoding="utf-8") as f:
    data.extend(json.load(f))
category_map = defaultdict(list)

for item in data:
    parts = item["id"].split("_")
    category = f"{parts[0]}_{parts[1]}"
    category_map[category].append(item)

train_set = []
test_set = []

train_ratio = 0.75
for cat, items in category_map.items():
    random.shuffle(items)
    split_idx = int(len(items) * train_ratio)
    if split_idx == len(items) and len(items) > 1:
        split_idx -= 1
    elif split_idx == 0 and len(items) > 0:
        split_idx = 1

    train_set.extend(items[:split_idx])
    test_set.extend(items[split_idx:])


with open("./data_split/new_train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=4, ensure_ascii=False)

with open("./data_split/new_test_data.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=4, ensure_ascii=False)

print(f"总样本数: {len(data)}")
print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")
print(f"包含类别数: {len(category_map.keys())}")
