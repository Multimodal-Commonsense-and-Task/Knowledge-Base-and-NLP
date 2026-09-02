import pickle
import logging
import argparse
import os
import ipdb
import openai
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import Counter

from synapse.envs.mind2web.env_utils import load_json
#from synapse.agents.mind2web_orig import eval_sample as eval_sample_orig
#from synapse.agents.mind2web_orig import eval_traj_sample as eval_traj_sample_orig

from synapse.agents.mind2web_plan import eval_sample as eval_sample_plan
from synapse.agents.mind2web_plan import eval_traj_sample as eval_traj_sample_plan

from synapse.agents.mind2web_plan import eval_synapse_orig as eval_traj_sample_orig
from synapse.agents.mind2web_plan import eval_synapse_new as eval_traj_sample_new

#os.environ["OPENAI_API_KEY"] = 'REDACTED' 
#os.environ["OPENAI_API_KEY"] = 'REDACTED'
#openai.api_key = os.environ["OPENAI_API_KEY"]


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    # 252, 177, 912
    parser.add_argument(
        "--benchmark", type=str, choices=["test_task", "test_website", "test_domain"]
    )
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=3)
    parser.add_argument("--see_previous_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_memory", action="store_true", default=False)
    parser.add_argument("--no_trajectory", action="store_true", default=False)

    parser.add_argument("--narrate", action="store_true", default=False)
    parser.add_argument("--comp", action="store_true", default=False)

    parser.add_argument("--plan", action="store_true", default=False)
    
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=-1)

    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--seed", type=int, default=0)

    # Ours
    parser.add_argument("--api", type=str, default="azure1", choices=["openai", "azure1", "azure2"])
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")


    parser.add_argument("--use_memory_annotation", action="store_true", default=False)
    parser.add_argument(
        "--annotated_memory_path", type=str, default=None, help="Path to annotated memory"
    )
    parser.add_argument("--mind2web_oracle", action="store_true", default=False, help="use oracle (base setting of mind2web) for generating thoughts")

    parser.add_argument("--planner_type", type=str, default="planning", choices=["planning", "heuristic"])

    parser.add_argument("--order_by_complexity", action="store_true", default=False)


    return parser


def main():
     
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    #handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
    logger.addHandler(handler)

    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "memory")
    #args.log_dir = os.path.join(current_path, "results/mind2web")

    if args.no_memory:
        logger.info(f"## Not using memory")
    else:
        logger.info(f"## Using memory")
    if args.no_trajectory:
        logger.info(f"## Not using trajectory")
    else:
        logger.info(f"## Using trajectory")
    
    # Ours
    if args.narrate:
        logger.info(f"## Using exemplar annotations")
    else:
        logger.info(f"## Not using exemplar annotations")    
    if args.comp:
        logger.info(f"## Using compositional annotations")
    else:
        logger.info(f"## Not using compositional annotations")
    if args.plan:
        logger.info(f"## Using plan")
    else:
        logger.info(f"## Not using plan")


    # Evaluate test set
    assert args.benchmark in ["test_task", "test_website", "test_domain"]
    samples = load_json(args.data_dir, args.benchmark)

    sample_infos = [{'len': len(item['actions']),'website': item['website'],'domain': item['domain'],}    for item in samples] 

    #ipdb.set_trace()

    start_idx = args.start_idx
    if args.n_samples > 0:
        end_idx = start_idx + args.n_samples
    else:
        end_idx = 912
    n_proc = 1
    chunk_size = n_proc
    
    #samples = samples[start_idx:end_idx]
    # randomly sample 20 samples that have length less than 12
    random.seed(args.seed)
    # get shuffled indices
    indices = list(range(len(samples)))
    #random.shuffle(indices)
    
    # Sort samples by length of 'confirmed_task'
    #indices = sorted(indices, key=lambda i: len(samples[i]['confirmed_task']), reverse=True)

    if args.order_by_complexity:
        logger.info(f"## Ordering by complexity")
        # sort samples by length of actions
        indices = sorted(indices, key=lambda i: len(samples[i]['actions']), reverse=True)

    #ipdb.set_trace()


    # NOTE get indices of samples that have length less than 12
    #indices = [i for i in indices if sample_infos[i]['len'] < 12]
    
    # get first n_samples indices
    indices = new_func(args, indices)
    
    #ipdb.set_trace()

    # get samples
    samples = [samples[i] for i in indices]
    
    #logger.info(f"## Randomly sampled {args.n_samples} samples that have length less than 12")
    logger.info(f"## Randomly sampled {args.n_samples} samples from the longest to the shortest")

    #ipdb.set_trace()


    #ipdb.set_trace()
    if args.start_idx > -1:
        # start from the specified index
        samples = samples[args.start_idx:]        
        logger.info(f"## Skipping {args.start_idx} samples. Evaluating {args.start_idx} to {args.start_idx + len(samples)}")


    skip_indices = []

    #print sample infos, including website, domain, and length distribution
    distribution = {'website': Counter(), 'domain': Counter(), 'length': Counter()}
    for i in indices:
        #print(f"{i}: {sample_infos[i]}")
        website = sample_infos[i]['website']
        domain = sample_infos[i]['domain']
        length = sample_infos[i]['len']
        distribution['website'][website] += 1
        distribution['domain'][domain] += 1
        distribution['length'][length] += 1
    logger.info(f"## Distribution:")
    logger.info(f"  Website: {distribution['website']}")
    logger.info(f"  Domain: {distribution['domain']}")
    logger.info(f"  Length: {distribution['length']}")

    
    #ipdb.set_trace()

    n = len(samples)
    logger.info(f"Number of samples: {n}")

    # add prediction scores and ranks to candidates
    with open(os.path.join(args.data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)
    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]
    for sample in samples:
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]

    """
    if args.narrate:
        #ipdb.set_trace()
        logger.info("## Using CoT Narrate")
        #eval_sample = eval_sample_narrate
        #eval_traj_sample = eval_traj_sample_narrate
        eval_sample = eval_sample_comp
        eval_traj_sample = eval_traj_sample_comp
    else:
        logger.info("## Using Original")
        eval_sample = eval_sample_orig
        eval_traj_sample = eval_traj_sample_orig

    if args.comp: # Override others
        logger.info("## Using Compositional annotations")
        eval_sample = eval_sample_comp
        eval_traj_sample = eval_traj_sample_comp
    """

    if args.plan:
        logger.info("## Using Plan")
        eval_sample = eval_sample_plan
        eval_traj_sample = eval_traj_sample_plan

        # tmp
        from synapse.agents.mind2web_plan import eval_trigger
        eval_traj_sample = eval_trigger

    else:
        logger.info("## Using Original")
        #eval_sample = eval_sample_orig
        #eval_traj_sample = eval_traj_sample_orig
        eval_traj_sample = eval_traj_sample_new
    #ipdb.set_trace()

    eval_func = eval_traj_sample if not args.no_trajectory else eval_sample
    with tqdm(total=n) as t:
        for i in range(0, n, chunk_size):
            chunk = samples[i : min(i + chunk_size, n)]
            indices_chunk = indices[i : min(i + chunk_size, n)]
            eval_func(i + start_idx, args, chunk[0])
            #ipdb.set_trace()

            #print(f"## Evaluating {i + start_idx} to {i + start_idx + len(chunk)}")
            logger.info(f"## Evaluating {indices_chunk}")

            if indices_chunk[0] in skip_indices:
                logger.info(f"## Skipping {indices_chunk[0]}")
                continue

            """
            with ThreadPoolExecutor(max_workers=n_proc) as executor:
                executor.map(
                    lambda p: eval_func(p[0], args, p[1]),
                    range(i + start_idx, i + start_idx + len(chunk)),
                    chunk,
                )
            """

            t.update(len(chunk))

def new_func(args, indices):
    indices = indices[:args.n_samples]
    return indices
    

if __name__ == "__main__":
    main()
