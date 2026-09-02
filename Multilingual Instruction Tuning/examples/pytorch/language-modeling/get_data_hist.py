from util.graph_utils import GraphMakerWithMatPlotLib
import numpy as np
graph_maker = GraphMakerWithMatPlotLib()

datas = []
for f in [
    "/data/open-lm-private_443/nabi/data/KO_dolly_deepl_15k.steps",
    "/data/open-lm-private_443/nabi/data/post-edit-lima.steps",
]:
    for l in open(f):
        data = float(l.split(',')[1])
        # print(data)
        if not np.isinf(data):
            datas.append(data)

print(sorted(datas))
graph_maker.add_histogram(datas)
graph_maker.write_fig('image.png')