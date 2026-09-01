import argparse
import glob
import json
from tqdm import tqdm

def merge_line(dataset : str, name : str):
    file_paths = sorted(glob.glob(f'data/{dataset}/{name}.*.jsonl'), key=lambda x: int(x.split('.')[-2]))
    with open(f'data/{dataset}/{name}.jsonl', 'w', encoding='utf-8') as fp_file:
        for file_path in tqdm(file_paths, desc='Merge line'):
            with open(file_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    line = line.strip()
                    fp_file.write(f'{line}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    merge_line(args.dataset, args.name)
