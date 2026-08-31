import logging
from tomt.benchmarks.terrier import TerrierRetriever

log = logging.getLogger(__name__)


def initialize_from_config(method, top_k, document_store, config_json):
    if method.startswith("terrier"):
        return TerrierRetriever(document_store, config_json, top_k=top_k)
    else:
        raise ValueError(method)
