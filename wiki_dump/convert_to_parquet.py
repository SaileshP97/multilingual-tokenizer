import os
import json
import pandas as pd

INPUT_DIR = "wiki_dumps"
OUTPUT_DIR = "wiki_parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for lang in [
    'hi', 'mr', 'sa', 'ne', 'mai', 'gu', 'bn', 'as', 'mni', 'bpy',
    'te', 'kn', 'ta', 'ml', 'ur', 'ks', 'sd', 'pa', 'or',
    'ps', 'my', 'awa', 'gom', 'sat'
]:
    input_dir = f"{INPUT_DIR}/{lang}"
    texts, urls = [], []

    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.startswith("wiki_"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            texts.append(data["text"])
                            urls.append(data["url"])
                        except Exception:
                            continue

    if texts:
        df = pd.DataFrame({"url": urls, "text": texts})
        os.makedirs(f"{OUTPUT_DIR}/{lang}", exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{lang}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} records for language '{lang}' → {out_path}")
    else:
        print(f"No data found for language '{lang}'")
