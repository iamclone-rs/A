import os

root_dir = "D:/Research/VLM_project/dataset/QuickDraw/photo"   # folder gốc chứa các class

tuberlin_classes = sorted([
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])

print(len(tuberlin_classes))
print(tuberlin_classes)
