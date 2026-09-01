import argparse
from hamu_tool.utils import CorpusReader

def build_index(dataset, file_name, idx_field):
    CorpusReader.build_index(data_path=f'data/{dataset}/{file_name}.jsonl', index_path=f'data/{dataset}/{file_name}.idx', idx_field=idx_field, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--idx_field', type=str)
    args = parser.parse_args()

    build_index(args.dataset, args.file_name, args.idx_field)
