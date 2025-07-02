import os
import pandas as pd
from tqdm import tqdm
import random

random.seed(13442)

lang_map = {
    'asm': 'as', 'ben': 'bn', 'guj': 'gu', 'kan': 'kn', 'kas': 'ks', 'mai': 'mai', 
    'mal': 'ml', 'mar': 'mr', 'mni': 'mni', 'nep': 'ne', 'ori': 'or', 'pan': 'pa', 
    'san': 'sa', 'sat': 'sat', 'snd': 'sd', 'tam': 'ta', 'tel': 'te', 'urd': 'ur'
}

lang_ratio = {
    "asm": 0.46, "awa": 100, "ben": 0.129, "bpy": 16.2, "brx": 41.66,
    "doi": 100, "gom": 20, "guj": 0.44, "kan": 0.43,
    "kas": 100, "mai": 2.52, "mal": 0.27, "mar": 0.252,
    "mni": 4, "my": 7.27, "nep": 0.023, "ori": 0.45,
    "pan": 0.35, "ps": 8.5, "san": 0.42, "sat": 7.6,
    "snd": 0.23, "tam": 0.2, "tel": 0.28, "urd": 0.08
}

sangraha_path = "./ai4bharat_sangraha/sangraha_data"
wiki_dump_path = "./wiki_dump/wiki_parquet"
final_data_path = "./training_data_parquet"

os.makedirs(final_data_path, exist_ok=True)

def english_char_percentage(text):
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    english_chars = sum(1 for c in text if c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz')
    return (english_chars / total_chars) * 100

if __name__ == "__main__":
    # First loop: Sangraha + Wiki
    # for lang in tqdm(['asm', 'ben', 'brx', 'doi', 'gom', 'guj', 
    #                   'kan', 'kas', 'mai', 'mal', 'mar', 'mni', 
    #                   'nep', 'ori', 'pan', 'san', 'sat', 'snd', 
    #                   'tam', 'tel', 'urd']):
        
    #     print(f"Processing {lang}...")
    #     sang_par = pd.read_parquet(f"{sangraha_path}/{lang}.parquet")

    #     lines = [sample.text for sample in sang_par.itertuples() if english_char_percentage(sample.text) < 2]
    #     random.shuffle(lines)
    #     text_blob = "\n".join(lines)[:2_000_000]  # limit by characters
    #     limited_lines = text_blob.splitlines()    # break back into lines

    #     df = pd.DataFrame({'text': limited_lines})
    #     df.to_parquet(f"{final_data_path}/{lang}.parquet", index=False)

    # # Second loop: Wiki-only languages not already processed
    # for lang in tqdm(['awa', 'bpy', 'my', 'ps']):
    #     print(f"Processing (wiki only) {lang}...")
    #     data = pd.read_parquet(f"{wiki_dump_path}/{lang}.parquet")
    #     lines = [sample.text for sample in data.itertuples() if english_char_percentage(sample.text) < 2]
    #     random.shuffle(lines)
    #     text_blob = "\n".join(lines)[:2_000_000]  # limit by characters
    #     limited_lines = text_blob.splitlines()    # break back into lines

    #     df = pd.DataFrame({'text': limited_lines})
    #     df.to_parquet(f"{final_data_path}/{lang}.parquet", index=False)

    all_dfs = []
    for file in os.listdir(final_data_path):
        if file.endswith(".parquet"):
            path = os.path.join(final_data_path, file)
            df = pd.read_parquet(path)
            df['lang'] = file.replace('.parquet', '')  # Optionally tag with language
            all_dfs.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Save the merged parquet file
    merged_df.to_parquet(os.path.join(final_data_path, "all_languages_merged.parquet"), index=False)

