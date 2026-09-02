import json
import argparse

def merge_data(data):
    new_data = []
    q_anchor = ''
    for i in range(len(data)):
        
        q = data[i]['question']
        if q != q_anchor:
            q_anchor = q
            new_data.append(data[i])
        else:
            new_data[-1]['ctxs'] += data[i]['ctxs']

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_score_path", type=str)
    parser.add_argument("--input_data_path", type=str)
    parser.add_argument("--output_path", type=str, default="../data/evaluator/")
    args = parser.parse_args()
    
    data_org = json.load(open(args.input_data_path))
    scores = json.load(open(args.input_score_path))

    assert len(data_org) == len(scores['score'])

    global_idx = 0
    for elem in data_org:
        for ctx in elem['ctxs']:
            ctx['r_score'] = scores['score'][global_idx]
            global_idx += 1

    # merge data
    data_merged = merge_data(data_org)
    
    
    # sort by r score
    for d in data_merged:
        d['ctxs'] = sorted(d['ctxs'], key=lambda x: x['r_score'], reverse=True)
        
        
        
    with open(args.output_path, 'w') as f:
        json.dump(data_merged, f, indent=2)

if __name__ == "__main__":

    main()