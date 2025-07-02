#!/bin/bash

# List of language codes
langs=(
  asm ben brx doi gom guj kan kas mai mal mar
  mni nep ori pan san sat snd tam tel urd
)

# Output directory
mkdir -p sangraha_data
cd sangraha_data || exit 1

# Loop through each language and download the file
for lang in "${langs[@]}"; do
  url="https://huggingface.co/datasets/ai4bharat/sangraha/resolve/main/verified/${lang}/data-0.parquet?download=true"
  echo "Downloading $lang..."
  wget -O "${lang}.parquet" "$url"
done

echo "All downloads complete."