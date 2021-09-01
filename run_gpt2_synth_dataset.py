# Modified from https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py

import argparse
import logging

import numpy as np
import json
import torch
import pdb
import random
import json
from tqdm import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_data(path):
    with open(path) as f:
        corpus = json.load(f)

    data = []
    for article in corpus['data']:
        paragraphs = article['paragraphs']
        for p in paragraphs:
            sents = [st.split()[:30] for st in p['context'].split('. ')]
            for sent in sents:
                if len(sent) < 20: continue # ignore short sequences
                half_id = len(sent) // 2
                prompt = ' '.join(sent[:half_id]).strip()
                ending = ' '.join(sent[half_id:-1]).strip()
                data.append([prompt, ending])

    print("----------------------------------------")
    print('[context]', data[0][0])
    print('[ending]', data[0][1])
    print('# data', len(data))
    print("----------------------------------------")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="train-v1.1.json",
        help="path to ground truth data",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2-medium",
        type=str,
        help="path to pre-trained model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=3, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args)

    data = get_data(args.data_path)
    random.shuffle(data)
    data = data[:30000]

    # Initialize the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    with open('SynGPT2_train.jsonl', "w", encoding="utf-8") as f:
        for idx, (prompt_text, gt) in enumerate(tqdm(data)):
            fields = {}
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            fields['ctx'] = tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)
            encoded_prompt = encoded_prompt.to(args.device)
            
            gt_ending = tokenizer.encode(gt)
            length = len(gt_ending)
            fields['label'] = 0
            fields['endings'] = [tokenizer.decode(gt_ending)] # avoid artifects created by the tokenizer
            
            # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                pad_token_id = pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )
            
            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()

                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                if text.find(tokenizer.eos_token) != -1:
                    text = text[: text.find(tokenizer.eos_token)]

                ending = text[len(fields['ctx']) :].strip()
                fields['endings'].append(ending)


            if idx == 0: # logging
                print(fields)
                print("----------------------------------------")
            
            f.write(json.dumps(fields) + '\n')
            f.flush()
        

if __name__ == "__main__":
    main()
