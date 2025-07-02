# multilingual-tokenizer

# 🧠 Indian Languages Text Corpus Processing Pipeline

This repository provides a complete pipeline to download, clean, merge, and prepare multilingual Indian language datasets from Sangraha and Wikipedia. It also includes tokenizer training and model initialization steps.

---

## 📁 Directory Structure

```bash
.
├── ai4bharat_sangraha/
│   ├── download_data.sh          # Downloads Sangraha corpus
│   └── sangraha_data/            # Contains downloaded Sangraha `.parquet` files
│
├── wiki_dump/
│   ├── download_wikidump.sh      # Downloads Wikipedia dumps in text form
│   ├── covert_to_parquet.py      # Converts raw Wiki text into `.parquet` format
│   └── wiki_parquet/             # Contains processed Wiki `.parquet` files
│
├── training_data/                # Output directory for cleaned and merged training data
│   └── all_languages_merged.parquet  # Final combined corpus
│
├── merge_training_data.py        # Filters and merges Sangraha + Wiki dumps into cleaned `.parquet` files
├── train_tokenizer.py            # Trains a tokenizer on the final text corpus
├── initialize_model.py           # Initializes a model using the trained tokenizer
├── run_pipeline.sh               # Main script to run the full pipeline
└── README.md                     # Project documentation (this file)

