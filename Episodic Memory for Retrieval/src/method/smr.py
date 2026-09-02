import argparse
import json
import logging

from alive_progress import alive_bar
from dataclasses import replace

from src.util.callback import Callback, DefaultIterativeCallback
from src.util.const import register_method
from src.retriever import Retriever
from src.util.dtype import Action, AgentOutput, Document, Query, RankedDocument, State
from src.util.llm import LLM

class Agent:
    def __init__(self, method : str, llm : str, init_temp : float = 0.1, port : int = 8000):
        self.method = method
        self.model = LLM(llm, port)
        with open(f'prompt/{method}.txt', 'r', encoding='utf-8') as fp:
            self.principle = fp.read().strip()
        self.docset = {}
        self.init_temp = init_temp
        self.temperature = init_temp

    def get_system_prompt(self) -> str:
        results = [self.principle]
        results.append('## Documents')
        for did in self.docset:
            doc = self.docset[did]
            results.append(f'[{doc.did}] {doc.text}')
        results = '\n'.join(results)
        return results

    def add_docset(self, current_state : State):
        self.docset = {}
        for rank in current_state.ranks:
            if rank.document.did not in self.docset:
                self.docset[rank.document.did] = rank.document

    def run(self, current_state : State) -> tuple[State, AgentOutput]:
        self.add_docset(current_state)
        dump = json.dumps({'query': current_state.query.text, 'ranks': [rank.document.did for rank in current_state.ranks]}, ensure_ascii=False)
        messages = [
            { 'role': 'system', 'content': self.get_system_prompt() },
            { 'role': 'user', 'content': f'{dump}' }
        ]
        while True:
            response, input_tokens, output_tokens = self.model.generate(messages=messages, remove_space=True, temperature=self.temperature)
            logging.debug(f'Input tokens: {input_tokens}, Output tokens: {output_tokens}')
            logging.debug(f'Messages: {json.dumps(messages, ensure_ascii=False)}')
            logging.debug(f'Response: {response}')
            output = self.validate_response(response)
            if output is None:
                self.temperature = min(self.temperature + 0.1, 0.8)
                continue
            self.temperature = self.init_temp
            output.input_tokens = input_tokens
            output.output_tokens = output_tokens
            match output.action:
                case Action.REFINE:
                    new_query = replace(current_state.query, text=output.query.strip())
                    current_state = replace(current_state, query=new_query)
                    break
                case Action.RERANK:
                    original_doc_map = { rank.document.did : rank for rank in current_state.ranks }
                    new_ranks = [original_doc_map[did] for did in output.ranks if did in original_doc_map]
                    remaining_ranks = [rank for rank in current_state.ranks if rank.document.did not in output.ranks]
                    new_ranks.extend(remaining_ranks)
                    current_state = replace(current_state, ranks=new_ranks)
                    break
                case Action.STOP:
                    break
        return current_state, output

    def validate_response(self, response : str) -> AgentOutput | None:
        agent_output = None
        try:
            response = response.strip().strip('`json').strip()
            output = json.loads(response)
            assert 'action' in output, 'Response must contain [action] field'
            assert 'reason' in output, 'Response must contain [reason] field'
            output['action'] = output['action'].strip().lower()
            output['reason'] = output['reason'].strip()
            assert output['action'] in ['refine', 'rerank', 'stop'], 'Action must be one of [refine, rerank, stop]'
            match output['action']:
                case 'refine':
                    assert 'query' in output, 'Response must contain [query] field when action is [refine]'
                    assert isinstance(output['query'], str), 'Query must be a string'
                    output['query'] = output['query'].strip()
                    assert len(output['query']) > 0, 'Query must not be empty'
                    agent_output = AgentOutput(action=Action.REFINE, reason=output['reason'], query=output['query'])
                case 'rerank':
                    assert 'ranks' in output, 'Response must contain [ranks] field when action is [rerank]'
                    assert isinstance(output['ranks'], list), 'Ranks must be a list'
                    for i in range(len(output['ranks'])):
                        assert isinstance(output['ranks'][i], str), 'Each document ID in ranks must be a string'
                        output['ranks'][i] = output['ranks'][i].strip()
                        assert len(output['ranks'][i]) > 0, 'Document ID in ranks must not be empty'
                    agent_output = AgentOutput(action=Action.RERANK, reason=output['reason'], ranks=output['ranks'])
                case 'stop':
                    agent_output = AgentOutput(action=Action.STOP, reason=output['reason'])
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f'Error: {e}')
            logging.error(f'Response: {response}')
            return None

        return agent_output

@register_method('smr')
class SMR:
    def __init__(self, args: argparse.Namespace, tag: str, callbacks : list[Callback] | None = None):
        self.args = args
        self.agent = Agent(method=args.method, llm=args.llm, init_temp=args.init_temp, port=args.port)
        retriever_kwargs = {'dataset': args.dataset}
        if args.retriever == 'bm25':
            retriever_kwargs.update({'k1': args.bm25_k1, 'b': args.bm25_b})
        self.retriever = Retriever(model_name=args.retriever, **retriever_kwargs)
        if callbacks is None:
            self.callbacks: list[Callback] = [DefaultIterativeCallback(dataset=args.dataset, query_type=args.query_type, file_name=f'{args.method}.{tag}.{args.idx}')]
        else:
            self.callbacks = callbacks

    def retrieve(self, query : Query, k : int = 10) -> tuple[list[RankedDocument], dict]:
        history = {'qid': query.qid, 'query': query.text, 'steps': []}
        results = self.retriever.retrieve(query, k=k + len(query.excluded_dids))
        results = [item for item in results if item.document.did not in query.excluded_dids][:k]
        current_state = State(query=query, ranks=results)
        for step in range(self.args.max_steps):
            current_state, agent_output = self.agent.run(current_state)
            match agent_output.action:
                case Action.REFINE:
                    results = self.retriever.retrieve(current_state.query, k=k + len(query.excluded_dids))
                    results = [item for item in results if item.document.did not in query.excluded_dids][:k]
                    current_dids = set([rank.document.did for rank in current_state.ranks])
                    remaining_ranks = [rank for rank in results if rank.document.did not in current_dids]
                    new_ranks = current_state.ranks + remaining_ranks
                    current_state = replace(current_state, ranks=new_ranks[:25])
                case Action.RERANK:
                    current_state = replace(current_state, ranks=current_state.ranks[:k])
            history['steps'].append({
                'step': step,
                'action': agent_output.action.name.lower(),
                'query': current_state.query.text,
                'ranks': [rank.document.did for rank in current_state.ranks],
                'reason': agent_output.reason,
                'input_tokens': agent_output.input_tokens,
                'output_tokens': agent_output.output_tokens,
            })
            for callback in self.callbacks:
                callback.on_step_end(step=step, query=current_state.query, state=current_state, agent_output=agent_output)
            if agent_output.action == Action.STOP:
                break
        final_ranks : list[RankedDocument] = []
        idx = 1
        for rank in current_state.ranks:
            final_ranks.append(RankedDocument(document=rank.document, score=360360 / idx))
            idx += 1
        history['ranks'] = [(rank.document.did, rank.score) for rank in final_ranks]
        for callback in self.callbacks:
            callback.on_query_end(query=query, ranks=final_ranks, history=history)
        return final_ranks, history

    def run(self, queries: list[Query], k : int = 10):
        with alive_bar(total=len(queries), title=f'Inference [{self.args.method.upper()}] on [{self.args.dataset}-{self.args.query_type}]') as bar:
            for query in queries:
                self.retrieve(query, k=k)
                bar()
        for callback in self.callbacks:
            callback.on_retrieval_end()
