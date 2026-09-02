import json
import pytrec_eval

from abc import ABC, abstractmethod

from src.util.const import EVAL_METRICS
from src.util.dtype import Query, State, RankedDocument, AgentOutput

class Callback(ABC):
    @abstractmethod
    def on_step_end(self, step: int, **kwargs):
        pass

    @abstractmethod
    def on_query_end(self, query: Query, ranks: list[RankedDocument], **kwargs):
        pass

    @abstractmethod
    def on_retrieval_end(self, **kwargs):
        pass

class DefaultIterativeCallback(Callback):
    def __init__(self, dataset: str, query_type: str, file_name: str):
        self.fp_history = open(f'output/history/{dataset}-{query_type}/{file_name}.jsonl', 'w', encoding='utf-8')
        self.fp_eval = open(f'output/evaluation/{dataset}-{query_type}/{file_name}.jsonl', 'w', encoding='utf-8')
        self.fp_eval_merged = open(f'output/evaluation/{dataset}-{query_type}/{file_name}.merged.jsonl', 'w', encoding='utf-8')
        self.metrics_all = {}

    def __del__(self):
        self.fp_history.close()
        self.fp_eval.close()
        self.fp_eval_merged.close()

    def on_step_end(self, step: int, **kwargs):
        agent_output = kwargs.get('agent_output')
        assert type(agent_output) is AgentOutput
        print(f'[Step {step}] Action : {agent_output.action}', flush=True)

    def on_query_end(self, query: Query, ranks: list[RankedDocument], history: dict):
        dump = json.dumps(history, ensure_ascii=False)
        self.fp_history.write(f'{dump}\n')

        qrel = {query.qid: {did: 1 for did in query.pos_dids}}
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, EVAL_METRICS)
        run = {query.qid: {rank.document.did: rank.score for rank in ranks}}
        metrics = evaluator.evaluate(run)
        dump = json.dumps({'qid': query.qid, 'metrics': metrics[query.qid]})
        self.fp_eval.write(f'{dump}\n')
        for metric in metrics[query.qid]:
            if metric not in self.metrics_all:
                self.metrics_all[metric] = []
            self.metrics_all[metric].append(metrics[query.qid][metric])

    def on_retrieval_end(self):
        for metric in self.metrics_all:
            self.metrics_all[metric] = 100 * sum(self.metrics_all[metric]) / len(self.metrics_all[metric])
        dump = json.dumps(self.metrics_all)
        self.fp_eval_merged.write(f'{dump}\n')

class DefaultRetrieverCallback(Callback):
    def __init__(self, dataset: str, query_type: str, file_name: str):
        self.fp_history = open(f'output/history/{dataset}-{query_type}/{file_name}.jsonl', 'w', encoding='utf-8')
        self.fp_eval = open(f'output/evaluation/{dataset}-{query_type}/{file_name}.jsonl', 'w', encoding='utf-8')
        self.fp_eval_merged = open(f'output/evaluation/{dataset}-{query_type}/{file_name}.merged.jsonl', 'w', encoding='utf-8')
        self.metrics_all = {}

    def on_step_end(self, agent_output: AgentOutput):
        pass

    def on_query_end(self, query: Query, ranks: list[RankedDocument]):
        dump = json.dumps({'qid': query.qid, 'query': query.text, 'ranks': [rank.document.did for rank in ranks]}, ensure_ascii=False)
        self.fp_history.write(f'{dump}\n')

        qrel = {query.qid: {did: 1 for did in query.pos_dids}}
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, EVAL_METRICS)
        run = {query.qid: {rank.document.did: rank.score for rank in ranks}}
        metrics = evaluator.evaluate(run)
        dump = json.dumps({'qid': query.qid, 'metrics': metrics[query.qid]})
        self.fp_eval.write(f'{dump}\n')
        for metric in metrics[query.qid]:
            if metric not in self.metrics_all:
                self.metrics_all[metric] = []
            self.metrics_all[metric].append(metrics[query.qid][metric])

    def on_retrieval_end(self):
        for metric in self.metrics_all:
            self.metrics_all[metric] = 100 * sum(self.metrics_all[metric]) / len(self.metrics_all[metric])
        dump = json.dumps(self.metrics_all)
        self.fp_eval_merged.write(f'{dump}\n')
