#!/usr/bin/env python3
"""
Entity Resolution Evaluation Script

Evaluates the multi-CRM account matching system using:
- Precision: What fraction of predicted matches are correct?
- Recall: What fraction of actual matches did we find?
- F1 Score: Harmonic mean of precision and recall

Compares rule-based vs ML-based approaches.

Usage:
    python scripts/evaluate_entity_resolution.py
    python scripts/evaluate_entity_resolution.py --n-accounts 500 --output reports/er_eval.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.entity_resolution import (
    EntityResolver,
    generate_synthetic_test_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(n_accounts: int = 200, output_path: str = None):
    """Run entity resolution evaluation"""

    logger.info("=" * 60)
    logger.info("Entity Resolution Evaluation")
    logger.info("=" * 60)

    # Generate test data
    logger.info(f"\nGenerating synthetic data ({n_accounts} accounts per org)...")
    accounts1, accounts2, labeled_pairs = generate_synthetic_test_data(
        n_accounts=n_accounts,
        match_rate=0.15,  # 15% of accounts have matches
        seed=42
    )

    logger.info(f"  Org 1 accounts: {len(accounts1)}")
    logger.info(f"  Org 2 accounts: {len(accounts2)}")
    logger.info(f"  Labeled pairs: {len(labeled_pairs)}")
    logger.info(f"  True matches: {labeled_pairs['is_match'].sum()}")

    # Split into train/test
    train_size = int(len(labeled_pairs) * 0.7)
    train_pairs = labeled_pairs.iloc[:train_size]
    test_pairs = labeled_pairs.iloc[train_size:]

    logger.info(f"\n  Train pairs: {len(train_pairs)}")
    logger.info(f"  Test pairs: {len(test_pairs)}")

    # Combine accounts for training/evaluation
    all_accounts = accounts1._append(accounts2, ignore_index=True)

    results = {}

    # Evaluate rule-based approach
    logger.info("\n" + "-" * 40)
    logger.info("Rule-Based Entity Resolution")
    logger.info("-" * 40)

    resolver = EntityResolver(match_threshold=0.7)

    # Test at different thresholds
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        resolver.match_threshold = threshold
        metrics = resolver.evaluate(test_pairs, all_accounts, threshold=threshold)
        logger.info(f"\nThreshold {threshold}:")
        logger.info(f"  Precision: {metrics.precision:.3f}")
        logger.info(f"  Recall:    {metrics.recall:.3f}")
        logger.info(f"  F1 Score:  {metrics.f1_score:.3f}")
        logger.info(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, FN: {metrics.false_negatives}")

        results[f'rule_based_t{threshold}'] = {
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'threshold': threshold
        }

    # Train and evaluate ML approach
    logger.info("\n" + "-" * 40)
    logger.info("ML-Based Entity Resolution")
    logger.info("-" * 40)

    resolver_ml = EntityResolver(match_threshold=0.7)
    train_metrics = resolver_ml.train(train_pairs, all_accounts)

    logger.info(f"\nTraining Results:")
    logger.info(f"  CV F1 Score: {train_metrics['cv_f1_mean']:.3f} (+/- {train_metrics['cv_f1_std']:.3f})")
    logger.info(f"  Training samples: {train_metrics['training_samples']}")

    for threshold in [0.5, 0.6, 0.7, 0.8]:
        metrics = resolver_ml.evaluate(test_pairs, all_accounts, threshold=threshold)
        logger.info(f"\nThreshold {threshold}:")
        logger.info(f"  Precision: {metrics.precision:.3f}")
        logger.info(f"  Recall:    {metrics.recall:.3f}")
        logger.info(f"  F1 Score:  {metrics.f1_score:.3f}")

        results[f'ml_based_t{threshold}'] = {
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'threshold': threshold
        }

    results['ml_training'] = train_metrics

    # Find best configurations
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    best_rule = max(
        [(k, v) for k, v in results.items() if k.startswith('rule_based')],
        key=lambda x: x[1]['f1_score']
    )
    best_ml = max(
        [(k, v) for k, v in results.items() if k.startswith('ml_based')],
        key=lambda x: x[1]['f1_score']
    )

    logger.info(f"\nBest Rule-Based: {best_rule[0]}")
    logger.info(f"  F1: {best_rule[1]['f1_score']:.3f}, P: {best_rule[1]['precision']:.3f}, R: {best_rule[1]['recall']:.3f}")

    logger.info(f"\nBest ML-Based: {best_ml[0]}")
    logger.info(f"  F1: {best_ml[1]['f1_score']:.3f}, P: {best_ml[1]['precision']:.3f}, R: {best_ml[1]['recall']:.3f}")

    improvement = (best_ml[1]['f1_score'] - best_rule[1]['f1_score']) / best_rule[1]['f1_score'] * 100
    logger.info(f"\nML improvement over rule-based: {improvement:+.1f}%")

    # Save report
    if output_path:
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'n_accounts_per_org': n_accounts,
                'n_labeled_pairs': len(labeled_pairs),
                'n_train': len(train_pairs),
                'n_test': len(test_pairs),
                'match_rate': float(labeled_pairs['is_match'].mean())
            },
            'results': results,
            'best_rule_based': {'config': best_rule[0], **best_rule[1]},
            'best_ml_based': {'config': best_ml[0], **best_ml[1]},
            'ml_improvement_pct': improvement
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nReport saved to: {output_path}")

    # Print reproduction info
    logger.info("\n" + "=" * 60)
    logger.info("REPRODUCTION")
    logger.info("=" * 60)
    logger.info("\nTo reproduce:")
    logger.info(f"  python scripts/evaluate_entity_resolution.py --n-accounts {n_accounts}")
    logger.info("\nDataset: Synthetic (15% match rate, seed=42)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity resolution")
    parser.add_argument(
        "--n-accounts",
        type=int,
        default=200,
        help="Number of accounts per org"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON report"
    )

    args = parser.parse_args()

    results = run_evaluation(args.n_accounts, args.output)

    # Print final summary table
    print("\n" + "=" * 70)
    print("ENTITY RESOLUTION METRICS")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 70)

    for key in sorted(results.keys()):
        if key == 'ml_training':
            continue
        val = results[key]
        print(f"{key:<25} {val['precision']:<12.3f} {val['recall']:<12.3f} {val['f1_score']:<12.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
