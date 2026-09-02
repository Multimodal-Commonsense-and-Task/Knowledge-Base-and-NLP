from typing import List, Optional, Dict, Callable, Union, Iterable
import hashlib
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
# We do not use nltk's get_ngrams since we handle token ID-based n-gram features via slicing.
# from nltk import ngrams as get_ngrams
import numpy as np

from data_selection.base import (
        DSIR,
        default_load_dataset_fn,
        default_parse_example_fn,
        _iterate_virtually_sharded_dataset,
)

from data_selection.utils import parallelize

wpt = WordPunctTokenizer()

def hash_buckets(text: str, num_buckets: int = 10000) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets

def get_ngram_counts_from_ids(token_ids, n: int = 2, num_buckets: int = 10000) -> np.ndarray:
    """Compute n-gram counts from a list of token IDs.

    Args:
        token_ids: List or 1D torch.Tensor of token IDs
        n: n in n-grams (default 2: unigram + bigram)
        num_buckets: Number of hash buckets

    Returns:
        Numpy array of size num_buckets, each bucket counting corresponding n-grams
    """
    # Convert to list if not already
    if not isinstance(token_ids, list):
        try:
            token_ids = token_ids.tolist()
        except Exception as e:
            raise TypeError("token_ids must be convertible to a list.") from e

    counts = np.zeros(num_buckets, dtype=int)
    for i in range(len(token_ids) - n + 1):
        ngram = tuple(token_ids[i:i + n])
        bucket = hash(ngram) % num_buckets
        counts[bucket] += 1
    return counts

class HashedNgramDSIR(DSIR):
    """DSIR with hashed n-gram features."""
    def __init__(self,
                 raw_datasets: List[str],
                 target_datasets: List[str],
                 cache_dir: str,
                 raw_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 raw_parse_example_fn: Callable[[Dict], list] = default_parse_example_fn,
                 target_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 target_parse_example_fn: Callable[[Dict], list] = default_parse_example_fn,
                 num_proc: Optional[int] = None,
                 ngrams: int = 2,
                 num_buckets: int = 10000,
                 tokenizer: str = 'wordpunct',
                 min_example_length: int = 100,
                 target_laplace_smoothing: float = 0.0,
                 separate_targets: bool = False,
                 target_proportions: Optional[List[float]] = None) -> None:
        """
        Args:
            raw_datasets: List of data paths
            target_datasets: List of data paths
            cache_dir: directory to store cached log importance weights
            raw_load_dataset_fn: function to load a dataset from a path
            raw_parse_example_fn: function that takes an example and returns it (here, token IDs)
            ngrams: N in N-grams. 2 means both unigram and bigrams.
            num_buckets: number of buckets to hash n-grams into.
            tokenizer: 'wordpunct' or 'word_tokenize' or 'gpt2'; here, if 'gpt2' is chosen,
                       we expect the data to be already tokenized as token IDs.
            min_example_length: minimum number of tokens in an example to be considered.
            target_laplace_smoothing: Smooth the target hashed ngram distribution. This parameter is a pseudo-count. This could be useful for small target datasets.
            separate_targets: whether to select data separately for each target and then join them
            target_proportions: weighting across multiple targets if separate_targets=True. Set to None to weight by the size of each target dataset
        """
        super().__init__(
            raw_datasets=raw_datasets,
            target_datasets=target_datasets,
            cache_dir=cache_dir,
            raw_load_dataset_fn=raw_load_dataset_fn,
            raw_parse_example_fn=raw_parse_example_fn,
            target_load_dataset_fn=target_load_dataset_fn,
            target_parse_example_fn=target_parse_example_fn,
            num_proc=num_proc,
            separate_targets=separate_targets,
            target_proportions=target_proportions)
        if tokenizer == 'word_tokenize':
            self.tokenizer = word_tokenize
        elif tokenizer == 'wordpunct':
            self.tokenizer = wpt.tokenize
        elif tokenizer == 'gpt2':
            # For GPT2, we assume the data is already a list of token IDs and use identity function.
            self.tokenizer = lambda x: x
        else:
            raise ValueError('tokenizer not recognized')
        self.ngrams = ngrams
        self.num_buckets = num_buckets
        self.min_example_length = min_example_length
        self.raw_probs = None
        self.target_probs = None
        self.log_diff = None
        self.target_laplace_smoothing = target_laplace_smoothing

    def featurizer(self, token_ids: list) -> np.ndarray:
        """Returns an n-gram feature vector from a list of token IDs."""
        # Compute n-gram counts directly from token_ids instead of plain text
        return get_ngram_counts_from_ids(token_ids, n=self.ngrams, num_buckets=self.num_buckets)

    def importance_estimator(self, features: np.ndarray) -> Union[float, np.ndarray]:
        return np.dot(self.log_diff, features)

    def get_perexample_metadata(self, ex: Dict, features: np.ndarray) -> int:
        """Returns token count (length) as per-example metadata."""
        remainder = self.ngrams * (self.ngrams - 1) / 2
        return (features.sum() + remainder) // self.ngrams

    def perexample_metadata_filter(self, concat_metadata: np.ndarray) -> np.array:
        """Filters examples with fewer than the minimum number of tokens."""
        return concat_metadata >= self.min_example_length

    def _fit_bow(self,
                 paths: List[str],
                 num_tokens_to_fit: Optional[int] = None,
                 load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 parse_example_fn: Callable[[Dict], list] = default_parse_example_fn) -> np.ndarray:

        sharded_datasets = self._get_virtually_sharded_datasets(paths)

        def job(args: Dict):
            path = args['path']
            num_shards = args['num_shards']
            shard_idx = args['shard_idx']

            counts = np.zeros(self.num_buckets, dtype=int)
            dataset = load_dataset_fn(path)
            iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
            for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                # Each example is already a list of token IDs
                token_ids = parse_example_fn(ex)
                counts += get_ngram_counts_from_ids(token_ids, n=self.ngrams, num_buckets=self.num_buckets)
                if num_tokens_to_fit is not None and counts.sum() > num_tokens_to_fit // len(sharded_datasets):
                    break
            return counts

        all_counts = parallelize(job, sharded_datasets, self.num_proc)
        counts = sum(all_counts)
        return counts

    def fit_importance_estimator(self, num_tokens_to_fit: Union[str, int] = 'auto') -> None:
        '''Fit the importance estimator.
        Args:
            num_tokens_to_fit: number of tokens to fit the raw dataset importance estimator on.
                               Set to "all" to fit on all tokens, and "auto" to determine
                               the number of tokens to fit on automatically (100k * num_buckets).
                               Set to an integer to fit on that many tokens.
        '''
        if num_tokens_to_fit == 'auto':
            num_tokens_to_fit = 100000 * self.num_buckets
        elif num_tokens_to_fit == 'all':
            num_tokens_to_fit = None

        self.raw_probs = self._fit_bow(
            self.raw_datasets,
            num_tokens_to_fit=num_tokens_to_fit,
            parse_example_fn=self.raw_parse_example_fn,
            load_dataset_fn=self.raw_load_dataset_fn)
        self.raw_probs = self.raw_probs / self.raw_probs.sum()

        if self.separate_targets:
            target_probs = []
            target_proportions = []
            for target_dataset in self.target_datasets:
                curr_target_probs = self._fit_bow(
                    [target_dataset],
                    num_tokens_to_fit=num_tokens_to_fit,
                    parse_example_fn=self.target_parse_example_fn,
                    load_dataset_fn=self.target_load_dataset_fn)
                target_proportions.append(curr_target_probs.sum())
                curr_target_probs = curr_target_probs + self.target_laplace_smoothing
                curr_target_probs = curr_target_probs / curr_target_probs.sum()
                target_probs.append(curr_target_probs)
            target_proportions = np.asarray(target_proportions)
            if self.target_proportions is None:
                self.target_proportions = target_proportions / target_proportions.sum()
            self.target_probs = np.asarray(target_probs)
        else:
            self.target_probs = self._fit_bow(
                self.target_datasets,
                num_tokens_to_fit=None,
                parse_example_fn=self.target_parse_example_fn,
                load_dataset_fn=self.target_load_dataset_fn)
            self.target_probs = self.target_probs + self.target_laplace_smoothing
            self.target_probs = self.target_probs / self.target_probs.sum()

        self.log_diff = np.log(self.target_probs + 1e-8) - np.log(self.raw_probs + 1e-8)