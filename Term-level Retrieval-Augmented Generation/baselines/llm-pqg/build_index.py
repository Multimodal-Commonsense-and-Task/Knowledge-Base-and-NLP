import argparse
from hamu_tool.utils import CorpusReader

def build_index(dataset, name, idx_field):
    CorpusReader.build_index(data_path=f'data/{dataset}/{name}.jsonl', index_path=f'data/{dataset}/{name}.idx', idx_field=idx_field, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--idx_field', type=str)
    args = parser.parse_args()

    build_index(args.dataset, args.name, args.idx_field)
