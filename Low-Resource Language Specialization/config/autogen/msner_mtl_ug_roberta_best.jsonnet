local t = import "DO_NOT_ERASE_pathfinder.libsonnet";

t.build_mtl_ner_ms("ug", "bert", ["roberta", "best"], "mtl_ug_roberta_best")
