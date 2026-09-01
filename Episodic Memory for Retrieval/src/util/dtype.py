from dataclasses import dataclass, field
from enum import Enum, auto

@dataclass(slots=True, frozen=True)
class Query:
    qid : str
    text : str
    gold_answer : str
    reasoning : str = ''
    pos_dids : set[str] = field(default_factory=set)
    pos_dids_long : set[str] = field(default_factory=set)
    excluded_dids : set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.qid or not self.text:
            raise ValueError('Query [qid] and [text] cannot be empty.')

@dataclass(slots=True, frozen=True)
class Document:
    did : str
    text : str

    def __post_init__(self):
        if not self.did or not self.text:
            raise ValueError('Document [did] and [text] cannot be empty.')

@dataclass(slots=True, frozen=True)
class RankedDocument:
    document: Document
    score: float

@dataclass(slots=True, frozen=True)
class State:
    query : Query
    ranks : list[RankedDocument] = field(default_factory=list)

class Action(Enum):
    REFINE = auto()
    RERANK = auto()
    STOP = auto()

@dataclass(slots=True)
class AgentOutput:
    action : Action
    reason : str
    query : str = ''
    ranks : list[str] = field(default_factory=list)
    input_tokens : int = 0
    output_tokens : int = 0
