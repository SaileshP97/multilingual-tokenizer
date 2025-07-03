#!/bin/bash

# List of language codes
lang_codes=('hi' 'mr' 'sa' 'ne' 'mai' 'gu' 'bn' 'as' 'mni' 'bpy' 'te' 'kn' 'as' 'mni' 'bpy' 'te' 'kn' 'ta' 'ml' 'ur' 'ks' 'sd' 'pa' 'or' 'en' 'ur' 'ps' 'my' 'sat' 'awa' 'gom')

# Base URL
base_url="https://dumps.wikimedia.org"

# Output directory
output_dir="./wiki_dumps"

# Create output directory
mkdir -p "$output_dir"
cd "$output_dir" || exit

# Loop over language codes
for lang in "${lang_codes[@]}"; do
    mkdir -p "$lang"
    cd "$lang" || exit

    echo "Downloading dumps for language: $lang"

    dump_url="${base_url}/${lang}wiki/latest/"

    # Use wget to mirror the dump directory, reject .xml files
    wget --recursive --no-parent --no-host-directories --cut-dirs=2 \
    --accept "pages-articles-multistream.xml.bz2" \
    --reject "index.html*, *.xml" "$dump_url"


    # Find the main article dump file (typically ends in 'pages-articles-multistream.xml.bz2')
    dump_file=$(find . -type f -name "*pages-articles*.xml.bz2" | head -n 1)

    if [[ -f "$dump_file" ]]; then
        echo "Running WikiExtractor on: $dump_file"

        # Create output directory for extracted text
        mkdir -p extracted

        python -m wikiextractor.WikiExtractor "$dump_file" -o extracted --json

        echo "Extraction complete for: $lang"
        find . -type f -name "*.bz2" -delete
        echo "Deleted .bz2 files"
    else
        echo "No valid dump file found for $lang"
    fi

    cd ..
done

echo "All downloads and extractions complete."