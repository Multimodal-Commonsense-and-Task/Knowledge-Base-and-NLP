import os
import re
import json
import time
import string
import numpy as np

from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from lcb_runner.evaluation import codegen_metrics
from utils.math_equivalence import is_equiv

from utils.rouge import Rouge
rouge = Rouge()
from utils.self_bleu import calculate_self_bleu

from transformers import AutoTokenizer



def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            # extracted_text = "No helpful information found."
            extracted_text = output
    elif mode in ['choose', 'qa']:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            # if mode in ['choose', 'qa']:
                # Handle 'choose' mode
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
        else:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', output) # last sentence
            if sentences:
                extracted_text = sentences[-1].strip()
            else:
                extracted_text = output
            # extracted_text = output
    else:
        extracted_text = output
    return extracted_text


def normalize_korean(text):
    text = okt.pos(text, stem=True)
    text = " ".join([word for word, tag in text if tag not in ['Josa', 'Punctuation']])
    return text

def deduplicate_sents(sents, do_normalize_korean=False):
    sents_set = set()
    sents_list = []
    if do_normalize_korean:
        sents = split_sentences(sents)
    else:
        sents = sents.split("\n")
    for sent in sents:
        for s in sent.split("\n"):
            if do_normalize_korean:
                s = normalize_korean(normalize_answer_qa(s))
            else:
                s = normalize_answer_qa(s)
            if len(s) == 0:
                continue
            if s in sents_set:
                continue
            sents_set.add(s)
            sents_list.append(s)
    return sents_list

def sent_tokenize(sents, do_normalize_korean=False):
    sents_set = set()
    sents_list = []
    if do_normalize_korean:
        sents = split_sentences(sents)
        sents = [normalize_korean(normalize_answer_qa(sent)) for sent in sents]
    else:
        sents = sents.split("\n")
    return sents

def tokenization(text):
    text = tokenizer.encode(text, add_special_tokens=False)
    text = " ".join([str(token_id) for token_id in text])
    return text

def truncate_tokens(text, domain=None):
    if domain is None:
        return text
    if domain == "explain":
        max_tokens = 822
    elif domain == "howto":
        max_tokens = 706
    elif domain == "recommend":
        max_tokens = 795
    elif domain == "short":
        max_tokens = 20480
    else:
        assert False
    
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[:max_tokens]
    text = tokenizer.decode(token_ids, add_special_tokens=False)
    return text

def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer, mode='gen', short_answer=[], domain=None, do_normalize_korean=False):
    final_metric = {
        "is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0,
        "rouge_1/f": 0, "rouge_1/r": 0, "rouge_1/p": 0,
        "rouge_2/r": 0, "rouge_2/f": 0, "rouge_2/p": 0,
        "rouge_l/f": 0, "rouge_l/r": 0, "rouge_l/p": 0,
        "rouge_lsum/f": 0, "rouge_lsum/r": 0, "rouge_lsum/p": 0,
        "bleu": 0, "meteor": 0, 'bert_score': 0,
    }
    output = str(output)
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        normalized_ground_truth_list = []
        for answer in labeled_answer:
            answer = str(answer)
            normalized_ground_truth = normalize_answer_qa(answer)
            normalized_ground_truth_list.append(normalized_ground_truth)

            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)
            for k in ["em", "acc"]:
                final_metric[k] = max(eval(k), final_metric[k])

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same > 0:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                final_metric["f1"] = max(eval("f1"), final_metric["f1"])
            
        rouge_l_f = rouge.calc_score([normalized_pred_answer], normalized_ground_truth_list)
        final_metric['rouge_l/f'] = max(final_metric['rouge_l/f'], rouge_l_f)
            
        rouge_l_f = rouge.calc_score([normalized_pred_answer], normalized_ground_truth_list)
        final_metric['rouge_l/f'] = max(final_metric['rouge_l/f'], rouge_l_f)

    elif mode == 'asqa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            answer = str(answer)
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            rouge_score = rouge([normalized_pred_answer], [normalized_ground_truth])
            for k in ["rouge_1", "rouge_2", "rouge_l"]:
                if final_metric[k+"/f"] < rouge_score[k+"/f"]:
                    final_metric[k+"/f"] = rouge_score[k+"/f"]
                    final_metric[k+"/r"] = rouge_score[k+"/r"]
                    final_metric[k+"/p"] = rouge_score[k+"/p"]
        str_ems = []
        for sublist in short_answer:
            normalized_short_answer = [normalize_answer_qa(item) for item in sublist]
            str_ems.append(max([s in normalized_pred_answer for s in normalized_short_answer]))
        str_em = np.mean(str_ems)
        final_metric["em"] = str_em

    else:
        labeled_answer = str(labeled_answer)
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer


def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False):
    if dataset_name == 'livecode':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []

        # Collect difficulty levels for per-domain metrics
        difficulties = []
        per_difficulty_count = {}
        num_valid_answer = 0

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            difficulty = item.get("difficulty", "Unknown")
            difficulties.append(difficulty)
            # Track metrics per domain
            if difficulty not in per_difficulty_count.keys():
                per_difficulty_count[difficulty] = 0

            pred_code = extract_answer(item['Output'], mode='codegen')
            if pred_code != '':
                num_valid_answer += 1
                per_difficulty_count[difficulty] += 1
            # Assuming each item has 'input_output' with 'inputs' and 'outputs'
            public_test_cases = json.loads(item.get("public_test_cases", "{}"))

            inputs, outputs = [], []
            for case in public_test_cases:
                inputs.append(case["input"])
                outputs.append(case["output"])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt


        # Call codegen_metrics with pass@1
        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1],  # Evaluate the top 1 generated result
            num_process_evaluate=2,   # Parallel evaluation
            timeout=10,  # Set timeout to 10 seconds
            debug=False,  # Enable debug mode
        )
        # print('samples_list', samples_list)
        # print('generations_list', generations_list)
        # print('metrics', metrics)

        # Extract pass@1
        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        # Initialize per-difficulty metrics
        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(difficulties):
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty].append(pass1)

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': pass_at_1,  # / num_valid_answer * len(input_list),
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }

        # Compute per-difficulty pass@1
        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            avg_pass = np.mean(passes) if len(passes) > 0 else 0.0
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                'pass@1': avg_pass,
                'num_valid_answer': f'{num_valid_answer} of {len(passes)}'
            }

        # Save the metrics
        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics
        }

    else:
        # Existing evaluation for other datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        avg_rouge1_f, avg_rouge1_p, avg_rouge1_r = [], [], []
        avg_rouge2_f, avg_rouge2_p, avg_rouge2_r = [], [], []
        avg_rougel_f, avg_rougel_p, avg_rougel_r = [], [], []
        num_valid_answer = 0

        # If the dataset is GPQA, track metrics per domain
        domain_metrics = {}

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            if dataset_name in ['gpqa', 'medmcqa'] or 'gpqa' in dataset_name:
                labeled_answer = item["answer"]
                # labeled_choice_answer = item["Correct Answer"]
                mode = 'choose'
            elif dataset_name in ['math500', 'aime', 'amc']:
                labeled_answer = item["answer"]
                mode = 'gen'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'crag', 'msmarco', 'msmarco_abs_500', 'msmarco_ext_500', 'msmarco_train_abs_500', 'msmarco_abs_500_per_doc', 
                                  'msmarco_train_abs_full', 'crag_500',
                                  'ambigqa',
                                  ] or "loong" in dataset_name:
                labeled_answer = item["answer"]
                mode = 'qa'
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                mode = 'choose'
            elif dataset_name == 'readable_doc':
                labeled_answer = ""
                mode = 'readable'
            elif 'asqa' in dataset_name:
                labeled_answer = item['answer']
                mode = 'asqa'
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            if 'asqa' in dataset_name:
                short_answer = item['short_answer']
            else:
                short_answer = []

            metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode,
                                                       short_answer=short_answer)
            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Input'] = input_prompt
            
            if dataset_name == "manualtc_all_qa":
                item["answerable"] = True if pred_answer == "답변 가능" else False

            # Determine the validity of the predicted answer
            my_method_valid = (pred_answer != '' and not 
                               (mode == 'choose' and (dataset_name == 'gpqa' or 'gpqa' in dataset_name)
                               and len(pred_answer) > 1))

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])

            avg_rouge1_f.append(metric['rouge_1/f'])
            avg_rouge1_p.append(metric['rouge_1/p'])
            avg_rouge1_r.append(metric['rouge_1/r'])
            avg_rouge2_f.append(metric['rouge_2/f'])
            avg_rouge2_p.append(metric['rouge_2/p'])
            avg_rouge2_r.append(metric['rouge_2/r'])
            avg_rougel_f.append(metric['rouge_l/f'])
            avg_rougel_p.append(metric['rouge_l/p'])
            avg_rougel_r.append(metric['rouge_l/r'])

            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

            # If the dataset is GPQA, attempt to track metrics per domain
            if dataset_name == 'gpqa' or 'gpqa' in dataset_name:
                domain = item.get("High-level domain", "Unknown")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['em'].append(metric['em'])
                domain_metrics[domain]['acc'].append(metric['acc'])
                domain_metrics[domain]['f1'].append(metric['f1'])
                domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                if my_method_valid:
                    domain_metrics[domain]['num_valid_answer'] += 1

        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        # Compute overall metrics
        overall_results = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'rouge_1': {
                'f': np.mean(avg_rouge1_f) if len(avg_rouge1_f) > 0 else 0.0,
                'p': np.mean(avg_rouge1_p) if len(avg_rouge1_p) > 0 else 0.0,
                'r': np.mean(avg_rouge1_r) if len(avg_rouge1_r) > 0 else 0.0,
            },
            'rouge_2': {
                'f': np.mean(avg_rouge2_f) if len(avg_rouge2_f) > 0 else 0.0,
                'p': np.mean(avg_rouge2_p) if len(avg_rouge2_p) > 0 else 0.0,
                'r': np.mean(avg_rouge2_r) if len(avg_rouge2_r) > 0 else 0.0,
            },
            'rouge_l': {
                'f': np.mean(avg_rougel_f) if len(avg_rougel_f) > 0 else 0.0,
                'p': np.mean(avg_rougel_p) if len(avg_rougel_p) > 0 else 0.0,
                'r': np.mean(avg_rougel_r) if len(avg_rougel_r) > 0 else 0.0,
            },
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }

        # If the dataset is GPQA, output average metrics per domain
        domain_avg_metrics = {}
        if dataset_name == 'gpqa' or 'gpqa' in dataset_name:
            for dm, m in domain_metrics.items():
                domain_avg_metrics[dm] = {
                    'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                    'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                    'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                    'rouge_1': {
                        'f': np.mean(avg_rouge1_f) if len(avg_rouge1_f) > 0 else 0.0,
                        'p': np.mean(avg_rouge1_p) if len(avg_rouge1_p) > 0 else 0.0,
                        'r': np.mean(avg_rouge1_r) if len(avg_rouge1_r) > 0 else 0.0,
                    },
                    'rouge_2': {
                        'f': np.mean(avg_rouge2_f) if len(avg_rouge2_f) > 0 else 0.0,
                        'p': np.mean(avg_rouge2_p) if len(avg_rouge2_p) > 0 else 0.0,
                        'r': np.mean(avg_rouge2_r) if len(avg_rouge2_r) > 0 else 0.0,
                    },
                    'rouge_l': {
                        'f': np.mean(avg_rougel_f) if len(avg_rougel_f) > 0 else 0.0,
                        'p': np.mean(avg_rougel_p) if len(avg_rougel_p) > 0 else 0.0,
                        'r': np.mean(avg_rougel_r) if len(avg_rougel_r) > 0 else 0.0,
                    },
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
                }

        # 保存总体和分domain的指标
        final_metrics = {'overall': overall_results}
        if dataset_name == 'gpqa' or 'gpqa' in dataset_name:
            final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday}.{t.tm_hour}h-{t.tm_min}m.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday}.{t.tm_hour}h-{t.tm_min}m.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {result_json_name}")
