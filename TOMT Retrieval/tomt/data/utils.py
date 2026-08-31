import json
import gzip
import pickle as pkl


def read_jsonl(file):
    with open(file) as reader:
        jj = []
        for line in reader:
            j = json.loads(line)
            jj.append(j)

        return jj


def read_qrels(file, for_pytrec=True):
    qrels = {}
    with open(file) as reader:
        for line in reader:
            (query_id, _, document_id, _) = line.split()
            if for_pytrec:
                qrels[query_id] = {
                    document_id: 1
                }
            else:
                qrels[query_id] = document_id
    return qrels


def write_jsonl(ll, path):
    with open(path, "w") as writer:
        for l in ll:
            writer.write(f"{json.dumps(l)}\n")


def write_json(d, path, indent=None, zipped=False):
    if zipped:
        with gzip.open(path, 'wt', encoding="ascii") as zipfile:
            json.dump(d, zipfile, indent=indent)
    else:
        with open(path, "w") as writer:
            json.dump(d, writer, indent=indent)


def read_json(path, zipped=False):
    if zipped:
        with gzip.open(path, 'rt', encoding="ascii") as zipfile:
            return json.load(zipfile)
    else:
        with open(path, "r") as reader:
            return json.load(reader)


def load_pickle(path):
    try:
        with open(path, "rb") as reader:
            return pkl.load(reader)
    except EOFError:
        raise EOFError(f"EOF Error for :{path}")


def write_pickle(obj, path):
    with open(path, "wb") as writer:
        pkl.dump(obj, writer)
