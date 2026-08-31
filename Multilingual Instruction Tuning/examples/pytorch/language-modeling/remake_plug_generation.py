import collections
import json
import sys
lang = 'es'

orig_file = f"/data/open-lm-private_443/nabi/taco-datasets/{lang}.jsonl"
orig_lines = iter(open(orig_file).readlines())
beam = False
mod_file = f"/data/open-lm-private_443/nabi/taco-datasets/{lang}_plug_iter1_greedy_outputreplaceonly_filtered{f'_beam' if beam else ''}.jsonl"
# mod_file = f"/data/open-lm-private_443/nabi/taco-datasets/{lang}_plug_iter2_greedy_outputreplaceonly_filtered{f'_beam' if beam else ''}.jsonl"

# https://stackoverflow.com/a/9079897
import re
def repetitions(s):
   r = re.compile(r"(.+?)\1+")
   for match in r.finditer(s):
       yield (match.group(1), len(match.group(0))/len(match.group(1)))

line_num = -1
with open(mod_file, 'w') as f:
    for split_num in range(100):
        for line in open(
                f"/data/open-lm-private_443/nabi/llamas/{lang}_70b_lora_bs1_ga2_plug/generation_{beam}_{split_num}.txt"):
        # for line in open(f"/data/open-lm-private_443/nabi/llamas/{lang}_70b_loraplug_iter1_greedy_outputreplaceonly_filtered_bs1_ga2_plug/generation_{beam}_{split_num}.txt"):
            orig_line = json.loads(next(orig_lines))
            line_num += 1
            try:
                en_output, ko_output = json.loads(line)['output'].split('### Response:\n')
                _, en_output = en_output.split('### English Response:\n')
                ratio = len(en_output) / len(ko_output)
                if not (0.2 < ratio < 5):
                    # print(en_output)
                    # print(ko_output)
                    continue
                counts = collections.defaultdict(int)
                for kw in ko_output.split():
                    counts[kw] += 1
                srted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                # srted = list(repetitions(ko_output))
                if srted[0][1] > 30:
                    # print(en_output)
                    # print(ko_output)
                    continue
                # if srted[0][1] > 20:
                #     print(en_output)
                #     print(ko_output)
                #     continue
            except:
                f.write(json.dumps(orig_line) + '\n')
                continue
            orig_line['output'] = ko_output
            orig_line['en_output'] = en_output
            f.write(json.dumps(orig_line) + '\n')