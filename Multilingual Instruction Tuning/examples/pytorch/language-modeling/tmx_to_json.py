from translate.storage.tmx import tmxfile
import json
from tqdm.auto import tqdm

with open("/data/open-lm-private_443/nabi/data/paracrawl_en-ko.tmx", 'rb') as fin:
	tmx_file = tmxfile(fin, 'en', 'ko')

with open("/data/open-lm-private_443/nabi/data/paracrawl_en-ko.jsonl", 'w') as f_w:
	for node in tmx_file.unit_iter():
		f_w.write(json.dumps(dict(
			en=node.source,
			ko=node.target
		)) + '\n')