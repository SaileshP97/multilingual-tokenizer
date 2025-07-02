import json
import copy
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers.models import BPE

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Tokenizer for Multi-lingual Text"
    )

    parser.add_argument(
        "--source_tokenizer",
        type=str,
        default="LingoIITGN/Ganga-2-1B",
        help="Source tokenizer",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="./training_data_parquet/all_languages_merged.parquet",
        help="File path for training.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="Multiligual_Tokenizer", help="Saving Dir"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)

    vocab = tokenizer.get_vocab()
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    merges = tokenizer_json["model"]["merges"]

    dataset = load_dataset("parquet", data_files=args.file_path)

    aux_tokenizer = tokenizer.train_new_from_iterator(
        dataset["train"]["text"],
        64000,
    )

    aux_tokenizer_json = json.loads(aux_tokenizer._tokenizer.to_str())
    aux_merges = aux_tokenizer_json["model"]["merges"]

    # merge the tokenizers
    # merge the tokenizers
    num_new_token = 0
    max_new_token = 32000
    ret_vocab = copy.copy(vocab)
    ret_merges = []
    old_merges = copy.copy(merges)

    for merge in aux_merges:
        # vocab
        [token_1, token_2] = merge

        if (len(token_1) > 20) or (len(token_2) > 20):
            continue

        token = token_1 + token_2
        if num_new_token < max_new_token:
            if token_1 not in ret_vocab and token_2 not in ret_vocab:  # both are new

                ret_vocab[token_1] = len(vocab) + num_new_token
                num_new_token += 1
                if token_1 != token_2:
                    ret_vocab[token_2] = len(vocab) + num_new_token
                    num_new_token += 1

            elif token_1 not in ret_vocab and token_2 in ret_vocab:  # new + old
                ret_vocab[token_1] = len(vocab) + num_new_token
                num_new_token += 1
            elif token_1 in ret_vocab and token_2 not in ret_vocab:  # old + new
                ret_vocab[token_2] = len(vocab) + num_new_token
                num_new_token += 1
            else:  # both are old
                pass
            if token not in ret_vocab:
                ret_vocab[token] = len(vocab) + num_new_token
                num_new_token += 1

        # merge
        if merge in merges:
            old_merges.remove(merge)
            ret_merges.append(merge)
        elif token in ret_vocab and token_1 in ret_vocab and token_2 in ret_vocab:
            ret_merges.append(merge)

    # Combine merges and convert to tuples
    final_merges = ret_merges + old_merges

    # Convert merge lists to tuples if they aren't already
    final_merges_tuples = []
    for merge in final_merges:
        if isinstance(merge, list):
            final_merges_tuples.append(tuple(merge))
        elif isinstance(merge, tuple):
            final_merges_tuples.append(merge)
        else:
            # Handle string format like "token1 token2"
            if isinstance(merge, str):
                parts = merge.split()
                if len(parts) == 2:
                    final_merges_tuples.append(tuple(parts))

    print(f"Total vocabulary size: {len(ret_vocab)}")
    print(f"Total merges: {len(final_merges_tuples)}")
    print(f"Added {num_new_token} new tokens")

    # retrain tokenizer
    tokenizer.backend_tokenizer.model = BPE(
        vocab=ret_vocab,
        merges=final_merges_tuples,  # Use tuples instead of lists
        fuse_unk=False,
        byte_fallback=True,
    )

    tokenizer.save_pretrained(args.output_dir)
