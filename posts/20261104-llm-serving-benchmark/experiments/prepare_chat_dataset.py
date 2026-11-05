"""Create OpenAI-format JSONL with exact template-inclusive input lengths.

Use the same tokenizer and default chat template as the server. Synthetic word
sequences test service performance, not response quality. No model is loaded.
"""
import argparse
import json
from pathlib import Path
import random
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--num-prompts', type=int, default=256)
    p.add_argument('--input-lengths',type=int,nargs='+',default=[128,1024,4096])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    words = [' apple',' book',' tree',' water',' green',' blue',' city',' river',
             ' music',' table',' house',' light',' road',' paper',' cloud',' garden']
    words = [w for w in words if len(tok.encode(w,add_special_tokens=False)) == 1]
    assert len(words) >= 4
    for length in args.input_lengths:
        rng = random.Random(42)
        path = args.output_dir / f'chat-{length}.jsonl'
        with path.open('w') as f:
            for i in range(args.num_prompts):
                prefix = f'{rng.getrandbits(64):016x}\n'
                tokens = rng.choices(words,k=length)
                def messages(n):
                    return [{'role':'user','content':prefix+''.join(tokens[:n])}]
                n = length - len(tok.apply_chat_template(messages(0), tokenize=True, add_generation_prompt=True,return_dict=False))
                data = messages(n)
                actual = len(tok.apply_chat_template(data,tokenize=True,add_generation_prompt=True,return_dict=False))
                assert actual == length, (length,i,actual)
                f.write(json.dumps({'messages':data,'max_tokens':128},ensure_ascii=False)+'\n')
        print(path, args.num_prompts, 'requests;', length, 'input tokens including chat template')


if __name__ == '__main__':
    main()
