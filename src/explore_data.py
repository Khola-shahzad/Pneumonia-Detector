# src/explore_data.py
import os

data_dir = '../chest_xray'

for split in ['train', 'val', 'test']:
    normal = len(os.listdir(os.path.join(data_dir, split, 'NORMAL')))
    pneumonia = len(os.listdir(os.path.join(data_dir, split, 'PNEUMONIA')))
    print(f"{split.upper()} — Normal: {normal}, Pneumonia: {pneumonia}")
