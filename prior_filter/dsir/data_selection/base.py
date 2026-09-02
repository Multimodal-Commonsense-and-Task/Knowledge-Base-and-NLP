import os
from typing import List, Optional, Dict, Callable, Iterable, Union
import multiprocessing as mp
from pathlib import Path
import shutil
import pickle
import json
import warnings
import glob
import pyarrow as pa
import pyarrow.ipc as ipc

import numpy as np
from tqdm import tqdm

from data_selection.utils import parallelize
from data_selection import __version__


def default_load_dataset_fn(path: str) -> Iterable[Dict]:
    """
    Loads a dataset composed of JSONL files or Apache Arrow file(s).
    If '*' is included in the path, the glob pattern is expanded to iterate over each file.
    For files ending with ".arrow", it first attempts ipc.open_file(),
    and falls back to ipc.open_stream() if that fails.
    """
    if '*' in path:
        file_list = sorted(glob.glob(path))
    else:
        file_list = [path]

    for file_path in file_list:
        if file_path.endswith(".arrow"):
            try:
                with pa.memory_map(file_path, 'r') as source:
                    try:
                        reader = ipc.open_file(source)
                    except pa.ArrowInvalid:
                        reader = ipc.open_stream(source)
                    table = reader.read_all()
                    for row in table.to_pylist():
                        yield row
            except Exception as e:
                print(f"Warning: Failed to read Arrow file {file_path}. Error: {e}")
                continue
        else:
            with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def default_parse_example_fn(ex: Dict) -> list:
    """
    Default parsing function.
    Returns the 'input_ids' field from each example.
    """
    if "input_ids" in ex:
        return ex["input_ids"]
    else:
        raise ValueError("Missing 'input_ids' field in example.")


def _iterate_virtually_sharded_dataset(dataset: Iterable, num_shards: int, shard_idx: int):
    for i, ex in enumerate(dataset):
        if i % num_shards == shard_idx:
            yield ex
    del dataset


class DSIR():
    """Base class for data selection with importance resampling (DSIR)."""
    __version__ = __version__

    def __init__(self,
                 raw_datasets: List[str],
                 target_datasets: List[str],
                 cache_dir: str,
                 raw_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 raw_parse_example_fn: Callable[[Dict], list] = default_parse_example_fn,
                 target_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 target_parse_example_fn: Callable[[Dict], list] = default_parse_example_fn,
                 num_proc: Optional[int] = None,
                 separate_targets: bool = False,
                 target_proportions: Optional[List[float]] = None) -> None:
        """
        Args:
            raw_datasets: List of data paths
            target_datasets: List of data paths
            cache_dir: Directory to store cached intermediates (log importance weights)
            raw_load_dataset_fn: Function to load raw dataset from path
            raw_parse_example_fn: Function that takes in an example and returns it (e.g., token IDs)
            target_load_dataset_fn: Function to load target dataset from path
            target_parse_example_fn: Function that takes in an example and returns it
            num_proc: number of CPUs to parallelize over. If None, use all available CPUs.
            separate_targets: whether to select data separately for each target and then join them
            target_proportions: weighting across multiple targets if separate_targets=True.
        """
        self.raw_datasets = raw_datasets
        self.target_datasets = target_datasets
        self.raw_parse_example_fn = raw_parse_example_fn
        self.raw_load_dataset_fn = raw_load_dataset_fn
        self.target_parse_example_fn = target_parse_example_fn
        self.target_load_dataset_fn = target_load_dataset_fn
        self.cache_dir = Path(cache_dir)
        if num_proc is None:
            try:
                # doesn't work on some systems
                self.num_proc = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_proc = mp.cpu_count()
        else:
            self.num_proc = num_proc
        self.log_importance_weights_dir = self.cache_dir / 'log_importance_weights'
        self.log_importance_weights_dir.mkdir(parents=True, exist_ok=True)
        self.perexample_metadata_dir = self.cache_dir / 'perexample_metadata'
        self.separate_targets = separate_targets
        self.target_proportions = target_proportions
        if self.target_proportions is not None:
            self.target_proportions = np.asarray(self.target_proportions) / np.sum(self.target_proportions)

    def _get_virtually_sharded_datasets(self, datasets: List[str]):
        """Return virtual shard parameters."""
        num_proc_per_shard = max(1, self.num_proc // len(datasets))
        if self.num_proc >= len(datasets):
            remainder = self.num_proc % len(datasets)
        else:
            remainder = 0

        overall_idx = 0
        shard_params = []
        for i, dataset in enumerate(datasets):
            curr_num_proc = num_proc_per_shard
            if i < remainder:
                curr_num_proc += 1
            for j in range(curr_num_proc):
                shard_params.append({'path': dataset, 'shard_idx': j, 'num_shards': curr_num_proc, 'overall_idx': overall_idx})
                overall_idx += 1
        return shard_params

    # Expected to be overridden in subclass (expects token ID list input)
    def featurizer(self, token_ids: list) -> np.ndarray:
        """Takes a token ID list and outputs a feature vector."""
        raise NotImplementedError

    def importance_estimator(self, features: np.ndarray) -> Union[float, np.ndarray]:
        """Takes a feature vector and outputs an importance weight."""
        raise NotImplementedError

    def get_perexample_metadata(self, ex: Dict, features: np.ndarray) -> np.ndarray:
        """Get per-example metadata.

        Args:
            ex: example (token ID list)
            features: feature vector
        """
        return NotImplementedError

    def fit_importance_estimator(self) -> None:
        """Fits parameters needed to run self.importance_estimator."""
        raise NotImplementedError

    def compute_importance_weights(self) -> None:

        def job(args: Dict):
            path = args['path']
            num_shards = args['num_shards']
            shard_idx = args['shard_idx']
            overall_idx = args['overall_idx']

            log_importance_weights = []
            perexample_metadata = []

            dataset = self.raw_load_dataset_fn(path)

            iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
            for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                # ex is a dictionary object using the "input_ids" field
                token_ids = ex.get("input_ids")
                if token_ids is None:
                    raise ValueError("Example is missing the 'input_ids' field.")
                features = self.featurizer(token_ids)
                log_importance_weights.append(self.importance_estimator(features))
                try:
                    perexample_metadata.append(self.get_perexample_metadata(ex, features))
                except NotImplementedError:
                    perexample_metadata = None

            log_importance_weights = np.asarray(log_importance_weights)
            save_path = Path(self.log_importance_weights_dir) / f"{overall_idx}.npy"
            np.save(str(save_path), log_importance_weights)
            if perexample_metadata is not None:
                self.perexample_metadata_dir.mkdir(parents=True, exist_ok=True)
                perexample_metadata = np.asarray(perexample_metadata)
                save_path = Path(self.perexample_metadata_dir) / f"{overall_idx}.npy"
                np.save(str(save_path), perexample_metadata)

        sharded_raw_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)
        parallelize(job, sharded_raw_datasets, self.num_proc)

    def perexample_metadata_filter(self, concat_metadata: np.ndarray) -> np.array:
        """Return a boolean array of examples that pass the filter according to the metadata."""
        return NotImplementedError

    def resample(self, out_dir: str, num_to_sample: int, cache_dir: str = None, top_k: bool = False) -> None:
        """Resample raw dataset according to importance weights.

        Args:
            out_dir (str): path to save resampled dataset
            num_to_sample (int): number of samples to resample
            cache_dir (str): path to cache resampled dataset
            top_k (bool): if True, get top_k examples by importance weight instead of sampling
        """
        if cache_dir is None:
            cache_dir = out_dir

        out_dir = Path(out_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        sharded_raw_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)

        # load log importance weights
        log_importance_weights_ls = [
            np.load(str(Path(self.log_importance_weights_dir) / f'{shard_params["overall_idx"]}.npy'), mmap_mode='r')
            for shard_params in sharded_raw_datasets]
        concat_log_importance_weights = np.concatenate(log_importance_weights_ls, axis=0)

        # filter examples by metadata first
        if Path(self.perexample_metadata_dir).exists():
            metadata_ls = [
                np.load(str(Path(self.perexample_metadata_dir) / f'{shard_params["overall_idx"]}.npy'), mmap_mode='r')
                for shard_params in sharded_raw_datasets]
            concat_metadata = np.concatenate(metadata_ls, axis=0)
            global_mask = self.perexample_metadata_filter(concat_metadata)
            del concat_metadata
        else:
            global_mask = np.ones(len(concat_log_importance_weights), dtype=bool)

        if self.separate_targets:
            # determine how many to sample per target
            num_to_sample_pertarget = [int(num_to_sample * p) for p in self.target_proportions]
            num_to_sample_pertarget[-1] += num_to_sample - sum(num_to_sample_pertarget)
        else:
            num_to_sample_pertarget = [num_to_sample]
            concat_log_importance_weights = concat_log_importance_weights[:, np.newaxis]

        chosen_mask = np.zeros(len(concat_log_importance_weights), dtype=bool)

        for i, curr_num_to_sample in enumerate(num_to_sample_pertarget):
            if curr_num_to_sample == 0:
                continue
            curr_log_importance_weights = concat_log_importance_weights[:, i]
            # apply filter
            curr_log_importance_weights = curr_log_importance_weights[global_mask]
            # noise the log_importance_weights (Gumbel top-k for sampling without replacement)
            if not top_k:
                curr_log_importance_weights += np.random.gumbel(size=curr_log_importance_weights.shape)

            # Take top-k
            nonzero_idxs = np.where(global_mask)[0]
            chosen_idxs = np.argpartition(-curr_log_importance_weights, curr_num_to_sample)[:curr_num_to_sample]
            chosen_idxs = nonzero_idxs[chosen_idxs]

            chosen_mask[chosen_idxs] = True
            # don't choose these examples again
            global_mask[chosen_idxs] = False

        del chosen_idxs
        del nonzero_idxs
        del concat_log_importance_weights
        del global_mask

        # split the global mask into per-dataset masks
        masks = []
        start_idx = 0
        for log_imp in log_importance_weights_ls:
            end_idx = start_idx + len(log_imp)
            masks.append(chosen_mask[start_idx:end_idx])
            start_idx = end_idx

        def job(args: Dict):
            in_path = args['in_path']
            out_path = args['out_path']
            mask = args['mask']
            shard_idx = args['shard_idx']
            num_shards = args['num_shards']
            global_start_idx = args['global_start_idx']

            # If input file is .arrow, process using pyarrow
            if in_path.endswith(".arrow"):
                try:
                    with pa.memory_map(in_path, 'r') as source:
                        try:
                            reader = ipc.open_file(source)
                        except pa.ArrowInvalid:
                            reader = ipc.open_stream(source)
                        table = reader.read_all()
                        rows = table.to_pylist()
                except Exception as e:
                    print(f"Warning: Failed to read Arrow file {in_path}. Error: {e}")
                    return

                shard_rows = [row for i, row in enumerate(rows) if i % num_shards == shard_idx]

                with open(out_path, 'w') as f:
                    for local_idx, row in enumerate(shard_rows):
                        if mask[local_idx]:
                            row["id"] = global_start_idx + local_idx
                            f.write(json.dumps(row) + '\n')
            else:
                if self.raw_load_dataset_fn.__name__ == 'default_load_dataset_fn':
                    curr_idx = 0
                    with open(out_path, 'w') as f:
                        with open(in_path, 'r') as f_in:
                            iterator = _iterate_virtually_sharded_dataset(f_in, num_shards, shard_idx)
                            for line in iterator:
                                if len(line.strip()) == 0:
                                    continue
                                if mask[curr_idx]:
                                    example = json.loads(line.strip())
                                    example["id"] = global_start_idx + curr_idx
                                    f.write(json.dumps(example) + '\n')
                                curr_idx += 1
                else:
                    dataset = self.raw_load_dataset_fn(in_path)
                    with open(out_path, 'w') as f:
                        iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
                        for i, ex in enumerate(iterator):
                            if mask[i]:
                                ex["id"] = global_start_idx + i
                                f.write(json.dumps(ex) + '\n')

        sharded_raw_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)
        job_args = []
        start_idx = 0
        for i, shard_params in enumerate(sharded_raw_datasets):
            end_idx = start_idx + len(log_importance_weights_ls[i])
            job_args.append({
                'out_path': cache_dir / f"{i}.jsonl",
                'in_path': shard_params['path'],
                'mask': masks[i],
                'shard_idx': shard_params['shard_idx'],
                'num_shards': shard_params['num_shards'],
                'global_start_idx': start_idx
            })
            start_idx = end_idx
        parallelize(job, job_args, self.num_proc)
        shutil.move(str(cache_dir), str(out_dir))

    def save(self, path: str) -> None:
        """Save parameters to save computation"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str, exclude_keys: Optional[List[str]] = None) -> None:
        """Load saved parameters.

        Args:
        path: path to saved parameters
        exclude_keys: keys to exclude from loading
        """

        with open(path, 'rb') as f:
            obj = pickle.load(f)

        if obj.__version__ != self.__version__:
            raise warnings.warn(f"Version mismatch: Saved version: {obj.__version__} != Current version: {self.__version__}")
        
        for k, v in obj.__dict__.items():
            if exclude_keys is not None and k in exclude_keys:
                continue
            setattr(self, k, v)