import json
import shutil
import os
import argparse

parser = argparse.ArgumentParser(description="Shrink a JSON dataset to a quarter of its size and create a backup.")
parser.add_argument('base', type=str, help="Base filename (e.g., 'rte' for data/rte.json)")
args = parser.parse_args()

base = args.base
src = os.path.join('data', f'{base}.json')
backup = os.path.join('data', f'{base}-backup.json')

# Step 1: Backup
shutil.copyfile(src, backup)
print(f"Backup created at {backup}")

# Step 2: Load JSON
with open(src, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 3: Shrink lists
for key, value in data.items():
    for subkey in value:
        if isinstance(value[subkey], list):
            n = len(value[subkey])
            new_len = max(1, n // 4) if n > 0 else 0
            value[subkey] = value[subkey][:new_len]

# Step 4: Save shrunk JSON
with open(src, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Shrunk data saved to {src}") 