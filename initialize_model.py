import math
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size


def instantiate_model_by_random(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    tie_word_embeddings: bool = False,
    causal_lm_model: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # expand the embeddings
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0),
        np.std(source_embeddings, axis=0),
        (
            round_to_nearest_multiple(len(target_tokenizer), 8),
            source_embeddings.shape[1],
        ),
    )
    target_embeddings[: source_embeddings.shape[0]] = source_embeddings

    if not tie_word_embeddings and causal_lm_model:
        print("You are using the output projection init.")
        source_head_embeddings = (
            source_model.get_output_embeddings().weight.detach().numpy()
        )
        target_head_embeddings = np.random.normal(
            np.mean(source_head_embeddings, axis=0),
            np.std(source_head_embeddings, axis=0),
            (
                round_to_nearest_multiple(len(target_tokenizer), 8),
                source_head_embeddings.shape[1],
            ),
        )
        target_head_embeddings[: source_head_embeddings.shape[0]] = (
            source_head_embeddings
        )

    # set weights
    target_model = source_model
    target_model.resize_token_embeddings(
        len(target_tokenizer),
        pad_to_multiple_of=8,  # See https://github.com/huggingface/transformers/issues/26303
    )
    target_model.get_input_embeddings().weight.data = torch.from_numpy(
        target_embeddings
    )
    target_model.config.vocab_size = round_to_nearest_multiple(len(target_tokenizer), 8)
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = torch.from_numpy(
            target_head_embeddings
        )
    else:
        target_model.tie_weights()

    return target_model, target_tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Tokenizer for Multi-lingual Text"
    )

    parser.add_argument(
        "--source_model", type=str, default="LingoIITGN/Ganga-2-1B", help="Source Model"
    )
    parser.add_argument(
        "--source_tokenizer",
        type=str,
        default="LingoIITGN/Ganga-2-1B",
        help="Source tokenizer",
    )
    parser.add_argument(
        "--target_tokenizer",
        type=str,
        default="./Multilingual_Ganga",
        help="Target tokenizer",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./Multilingual_Ganga", help="Saving Dir"
    )
    parser.add_argument(
        "--causal_lm_model",
        action="store_true",
        help="Whether to initialize language head or not.",
    )
    args = parser.parse_args()

    source_model = AutoModelForCausalLM.from_pretrained(args.source_model)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)

    target_model, target_tokenizer = instantiate_model_by_random(
        source_model=source_model,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        causal_lm_model=args.causal_lm_model,
    )

    target_tokenizer.save_pretrained(args.output_dir)
    target_model.save_pretrained(args.output_dir)
    print(f"Model saved at {args.output_dir}")
