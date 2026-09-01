#!/usr/bin/env python3
"""
Evaluation script for EXIT model.
Performs batch evaluation on test dataset and generates classification metrics.
"""

import os
import argparse
import logging
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer
from peft import PeftModel, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ExitEvaluator:
    """Evaluator class for EXIT model."""
    
    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        cache_dir: str = "./cache"
    ):
        """Initialize the evaluator with model paths."""
        logger.info(f"Loading base model from {base_model_path}")
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=cache_dir
        )
        
        # Load model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()
    
    def generate_prompt(
        self,
        query: str,
        context: str,
        sentence: str
    ) -> str:
        """Generate prompt for model input."""
        return (
            f"<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{context}\n"
            f"Sentence:\n{sentence}\n"
            f'Is this sentence useful in answering the query? Answer only "Yes" or "No".'
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str],
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Predict relevance for a batch of sentences."""
        prompts = [
            self.generate_prompt(q, c, s)
            for q, c, s in zip(queries, contexts, sentences)
        ]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        # Get logits for Yes/No tokens
        yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        
        logits = outputs.logits[:, -1, [yes_id, no_id]]
        probs = torch.softmax(logits, dim=-1)
        
        predictions = ["Yes" if p[0] > p[1] else "No" for p in probs]
        
        return predictions, probs

def evaluate_model(
    base_model_path: str,
    checkpoint_path: str,
    test_data: Dataset,
    threshold: float = 0.5,
    batch_size: int = 16
) -> Dict[str, any]:
    """
    Evaluate model on test dataset.
    
    Args:
        base_model_path: Path to base model
        checkpoint_path: Path to trained checkpoint
        test_data: Test dataset
        threshold: Confidence threshold for predictions
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    evaluator = ExitEvaluator(base_model_path, checkpoint_path)
    
    y_true = []
    y_pred = []
    
    logger.info("Starting evaluation...")
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data.select(range(i, min(i + batch_size, len(test_data))))
        
        predictions, probs = evaluator.predict_batch(
            batch['query'],
            batch['full_passage'],
            batch['sentence_text']
        )
        
        # Apply threshold
        predictions = [
            'Yes' if prob[0] >= threshold else 'No'
            for prob in probs
        ]
        
        y_true.extend(batch['label'])
        y_pred.extend(predictions)
    
    # Compute metrics
    class_report = classification_report(
        y_true,
        y_pred,
        labels=['Yes', 'No'],
        output_dict=True
    )
    conf_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=['Yes', 'No']
    )
    
    return {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'y_true': y_true,
        'y_pred': y_pred
    }

def save_results(
    results: Dict[str, any],
    output_dir: str,
    dataset_name: str,
    threshold: float
) -> None:
    """Save evaluation results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}_eval_threshold_{threshold:.2f}.txt"
    )
    
    with open(output_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Threshold: {threshold}\n\n")
        
        f.write("Classification Report:\n")
        report = classification_report(
            results['y_true'],
            results['y_pred'],
            labels=['Yes', 'No']
        )
        f.write(report)
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(results['confusion_matrix']))
        f.write("\n")
    
    logger.info(f"Results saved to {output_file}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate EXIT model")
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-2b-it",
        help="Base model path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        # Load test data
        logger.info(f"Loading test data from {args.test_dataset}")
        test_data = load_from_disk(args.test_dataset)
        
        # Run evaluation
        results = evaluate_model(
            args.base_model,
            args.checkpoint,
            test_data,
            args.threshold,
            args.batch_size
        )
        
        # Save results
        dataset_name = os.path.basename(os.path.dirname(args.test_dataset))
        save_results(
            results,
            args.output_dir,
            dataset_name,
            args.threshold
        )
        
        # Print summary
        logger.info("\nEvaluation Results:")
        logger.info(f"F1 Score (Yes): {results['classification_report']['Yes']['f1-score']:.4f}")
        logger.info(f"F1 Score (No): {results['classification_report']['No']['f1-score']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()