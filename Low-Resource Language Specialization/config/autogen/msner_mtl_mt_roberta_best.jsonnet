local t = import "DO_NOT_ERASE_pathfinder.libsonnet";

t.build_mtl_ner_ms("mt", "bert", ["roberta", "best"], "mtl_mt_roberta_best")
