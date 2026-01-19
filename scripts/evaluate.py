#!/usr/bin/env python3
"""
ML Model Evaluation Script for Cross-Sell Recommendation System

Evaluates the recommendation model using proper ranking metrics:
- Precision@K, Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage (accounts receiving recommendations)
- Compares against baselines (random, popularity, similarity-only)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --full --output reports/evaluation_report.json
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.pipeline import (
    EnsembleScorer,
    FeatureEngineering,
    ModelConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    coverage: float
    total_recommendations: int
    relevant_items: int


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    timestamp: str
    dataset_info: Dict
    model_metrics: EvaluationMetrics
    baseline_random: EvaluationMetrics
    baseline_popularity: EvaluationMetrics
    baseline_similarity: EvaluationMetrics
    improvement_over_random: Dict[str, float]
    improvement_over_popularity: Dict[str, float]


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Calculate Discounted Cumulative Gain at K"""
    relevances = np.asarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K"""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevances: np.ndarray, k: int) -> float:
    """Calculate Precision at K"""
    return np.mean(np.asarray(relevances)[:k]) if len(relevances) >= k else 0.0


def recall_at_k(relevances: np.ndarray, k: int, total_relevant: int) -> float:
    """Calculate Recall at K"""
    if total_relevant == 0:
        return 0.0
    return np.sum(np.asarray(relevances)[:k]) / total_relevant


def generate_synthetic_evaluation_data(
    n_accounts: int = 500,
    n_products: int = 50,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data for evaluation.

    Returns:
        accounts_df: Account features
        interactions_df: Historical interactions (ground truth)
        candidates_df: Candidate pairs to score
    """
    np.random.seed(seed)

    industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
    countries = ["USA", "UK", "Germany", "France", "Canada"]

    # Generate accounts
    accounts = []
    for i in range(n_accounts):
        accounts.append({
            "Id": f"ACC_{i:04d}",
            "Name": f"Company_{i}",
            "Industry": np.random.choice(industries),
            "AnnualRevenue": np.random.lognormal(15, 1.5),
            "NumberOfEmployees": int(np.random.lognormal(5, 1.5)),
            "BillingCountry": np.random.choice(countries),
            "CreatedDate": pd.Timestamp("2020-01-01") + pd.Timedelta(days=np.random.randint(0, 1000)),
            "LastActivityDate": pd.Timestamp("2024-01-01") - pd.Timedelta(days=np.random.randint(0, 365)),
            "Type": np.random.choice(["Customer", "Prospect", "Partner"], p=[0.5, 0.3, 0.2]),
        })
    accounts_df = pd.DataFrame(accounts)

    # Generate ground truth interactions (which accounts have actually bought cross-sell)
    # This simulates historical conversion data
    interactions = []
    for i in range(n_accounts):
        for j in range(i + 1, n_accounts):
            # Create some signal: same industry + similar size = higher conversion probability
            acc1, acc2 = accounts[i], accounts[j]

            base_prob = 0.02  # 2% base conversion rate

            # Industry match bonus
            if acc1["Industry"] == acc2["Industry"]:
                base_prob += 0.05

            # Size compatibility bonus
            size_ratio = min(acc1["NumberOfEmployees"], acc2["NumberOfEmployees"]) / \
                        max(acc1["NumberOfEmployees"], acc2["NumberOfEmployees"])
            base_prob += 0.03 * size_ratio

            # Geographic proximity bonus
            if acc1["BillingCountry"] == acc2["BillingCountry"]:
                base_prob += 0.02

            # Type compatibility
            if acc1["Type"] == "Customer" and acc2["Type"] == "Customer":
                base_prob += 0.03

            # Randomly determine if conversion happened
            if np.random.random() < base_prob:
                interactions.append({
                    "account1_id": acc1["Id"],
                    "account2_id": acc2["Id"],
                    "converted": 1,
                    "conversion_value": np.random.lognormal(10, 1),
                })

    interactions_df = pd.DataFrame(interactions)

    # Generate candidate pairs (all pairs, not just converted ones)
    candidates = []
    for i in range(n_accounts):
        for j in range(i + 1, min(i + 50, n_accounts)):  # Limit for performance
            acc1, acc2 = accounts[i], accounts[j]
            converted = interactions_df[
                ((interactions_df["account1_id"] == acc1["Id"]) &
                 (interactions_df["account2_id"] == acc2["Id"])) |
                ((interactions_df["account1_id"] == acc2["Id"]) &
                 (interactions_df["account2_id"] == acc1["Id"]))
            ]
            candidates.append({
                "account1_id": acc1["Id"],
                "account2_id": acc2["Id"],
                "label": 1 if len(converted) > 0 else 0,
            })

    candidates_df = pd.DataFrame(candidates)

    logger.info(f"Generated {len(accounts_df)} accounts, {len(interactions_df)} conversions, "
                f"{len(candidates_df)} candidate pairs")
    logger.info(f"Positive rate: {candidates_df['label'].mean():.2%}")

    return accounts_df, interactions_df, candidates_df


def create_features_for_pair(
    acc1: pd.Series,
    acc2: pd.Series,
    feature_engineer: FeatureEngineering
) -> np.ndarray:
    """Create feature vector for an account pair"""
    return feature_engineer.create_cross_org_features(acc1, acc2)


def evaluate_model(
    scorer: EnsembleScorer,
    feature_engineer: FeatureEngineering,
    accounts_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    model_name: str = "model"
) -> EvaluationMetrics:
    """Evaluate a model using ranking metrics"""

    # Score all candidates
    scores = []
    for _, row in candidates_df.iterrows():
        acc1 = accounts_df[accounts_df["Id"] == row["account1_id"]].iloc[0]
        acc2 = accounts_df[accounts_df["Id"] == row["account2_id"]].iloc[0]

        features = create_features_for_pair(acc1, acc2, feature_engineer)

        if scorer.is_trained:
            score, _ = scorer.predict(features.reshape(1, -1))
            scores.append(score[0])
        else:
            scores.append(0.5)  # Default score if not trained

    candidates_df = candidates_df.copy()
    candidates_df["score"] = scores

    # Sort by score (descending) to get ranking
    ranked = candidates_df.sort_values("score", ascending=False)
    relevances = ranked["label"].values

    total_relevant = candidates_df["label"].sum()

    # Calculate metrics at different K values
    metrics = EvaluationMetrics(
        precision_at_5=precision_at_k(relevances, 5),
        precision_at_10=precision_at_k(relevances, 10),
        precision_at_20=precision_at_k(relevances, 20),
        recall_at_5=recall_at_k(relevances, 5, total_relevant),
        recall_at_10=recall_at_k(relevances, 10, total_relevant),
        recall_at_20=recall_at_k(relevances, 20, total_relevant),
        ndcg_at_5=ndcg_at_k(relevances, 5),
        ndcg_at_10=ndcg_at_k(relevances, 10),
        ndcg_at_20=ndcg_at_k(relevances, 20),
        coverage=len(candidates_df[candidates_df["score"] > 0.5]) / len(candidates_df),
        total_recommendations=len(candidates_df),
        relevant_items=int(total_relevant),
    )

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Precision@10: {metrics.precision_at_10:.4f}")
    logger.info(f"  Recall@10: {metrics.recall_at_10:.4f}")
    logger.info(f"  NDCG@10: {metrics.ndcg_at_10:.4f}")
    logger.info(f"  Coverage: {metrics.coverage:.2%}")

    return metrics


class RandomBaseline:
    """Random baseline scorer"""
    is_trained = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return np.random.random(len(X)), {}


class PopularityBaseline:
    """Popularity-based baseline (favors larger accounts)"""
    is_trained = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Use first feature (typically revenue-related) as proxy for popularity
        scores = 1 / (1 + np.exp(-X[:, 0])) if X.shape[1] > 0 else np.ones(len(X)) * 0.5
        return scores, {}


class SimilarityBaseline:
    """Similarity-only baseline (uses feature similarity)"""
    is_trained = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Sum of features as similarity proxy
        scores = np.sum(X, axis=1)
        # Normalize to 0-1
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores, {}


def run_evaluation(full: bool = False, output_path: Optional[str] = None) -> EvaluationReport:
    """Run complete evaluation pipeline"""

    logger.info("=" * 60)
    logger.info("Cross-Sell Recommendation Model Evaluation")
    logger.info("=" * 60)

    # Generate evaluation data
    n_accounts = 500 if full else 200
    accounts_df, interactions_df, candidates_df = generate_synthetic_evaluation_data(
        n_accounts=n_accounts,
        seed=42
    )

    # Split into train/test (time-based split simulation)
    train_candidates, test_candidates = train_test_split(
        candidates_df, test_size=0.3, random_state=42, stratify=candidates_df["label"]
    )

    logger.info(f"\nDataset Split:")
    logger.info(f"  Train: {len(train_candidates)} pairs ({train_candidates['label'].mean():.2%} positive)")
    logger.info(f"  Test: {len(test_candidates)} pairs ({test_candidates['label'].mean():.2%} positive)")

    # Prepare training data
    feature_engineer = FeatureEngineering()

    X_train = []
    y_train = []
    for _, row in train_candidates.iterrows():
        acc1 = accounts_df[accounts_df["Id"] == row["account1_id"]].iloc[0]
        acc2 = accounts_df[accounts_df["Id"] == row["account2_id"]].iloc[0]
        features = create_features_for_pair(acc1, acc2, feature_engineer)
        X_train.append(features)
        y_train.append(row["label"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train the ensemble model
    logger.info("\nTraining Ensemble Model...")
    config = ModelConfig(nn_epochs=20 if not full else 50)  # Faster for quick eval
    scorer = EnsembleScorer(config)
    training_metrics = scorer.train(X_train, y_train)

    logger.info(f"Training Metrics: {training_metrics}")

    # Evaluate all models
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results on Test Set")
    logger.info("=" * 60)

    model_metrics = evaluate_model(
        scorer, feature_engineer, accounts_df, test_candidates, "Ensemble Model"
    )

    random_metrics = evaluate_model(
        RandomBaseline(), feature_engineer, accounts_df, test_candidates, "Random Baseline"
    )

    popularity_metrics = evaluate_model(
        PopularityBaseline(), feature_engineer, accounts_df, test_candidates, "Popularity Baseline"
    )

    similarity_metrics = evaluate_model(
        SimilarityBaseline(), feature_engineer, accounts_df, test_candidates, "Similarity Baseline"
    )

    # Calculate improvements
    def calc_improvement(model_val: float, baseline_val: float) -> float:
        if baseline_val == 0:
            return 0.0
        return ((model_val - baseline_val) / baseline_val) * 100

    improvement_random = {
        "ndcg_at_10": calc_improvement(model_metrics.ndcg_at_10, random_metrics.ndcg_at_10),
        "precision_at_10": calc_improvement(model_metrics.precision_at_10, random_metrics.precision_at_10),
        "recall_at_10": calc_improvement(model_metrics.recall_at_10, random_metrics.recall_at_10),
    }

    improvement_popularity = {
        "ndcg_at_10": calc_improvement(model_metrics.ndcg_at_10, popularity_metrics.ndcg_at_10),
        "precision_at_10": calc_improvement(model_metrics.precision_at_10, popularity_metrics.precision_at_10),
        "recall_at_10": calc_improvement(model_metrics.recall_at_10, popularity_metrics.recall_at_10),
    }

    # Create report
    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        dataset_info={
            "n_accounts": len(accounts_df),
            "n_candidates": len(candidates_df),
            "n_train": len(train_candidates),
            "n_test": len(test_candidates),
            "positive_rate": float(candidates_df["label"].mean()),
        },
        model_metrics=model_metrics,
        baseline_random=random_metrics,
        baseline_popularity=popularity_metrics,
        baseline_similarity=similarity_metrics,
        improvement_over_random=improvement_random,
        improvement_over_popularity=improvement_popularity,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\nModel vs Random Baseline:")
    logger.info(f"  NDCG@10 improvement: {improvement_random['ndcg_at_10']:+.1f}%")
    logger.info(f"  Precision@10 improvement: {improvement_random['precision_at_10']:+.1f}%")

    logger.info(f"\nModel vs Popularity Baseline:")
    logger.info(f"  NDCG@10 improvement: {improvement_popularity['ndcg_at_10']:+.1f}%")
    logger.info(f"  Precision@10 improvement: {improvement_popularity['precision_at_10']:+.1f}%")

    # Save report if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "dataset_info": report.dataset_info,
            "model_metrics": asdict(report.model_metrics),
            "baseline_random": asdict(report.baseline_random),
            "baseline_popularity": asdict(report.baseline_popularity),
            "baseline_similarity": asdict(report.baseline_similarity),
            "improvement_over_random": report.improvement_over_random,
            "improvement_over_popularity": report.improvement_over_popularity,
        }

        with open(output_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"\nReport saved to: {output_path}")

    # Print reproduction instructions
    logger.info("\n" + "=" * 60)
    logger.info("REPRODUCTION")
    logger.info("=" * 60)
    logger.info("\nTo reproduce these results:")
    logger.info("  python scripts/evaluate.py --full --output reports/evaluation_report.json")
    logger.info("\nDataset: Synthetic (500 accounts, ~12k candidate pairs)")
    logger.info("Split: 70/30 train/test, stratified by label")
    logger.info("Seed: 42 (fixed for reproducibility)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-sell recommendation model")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation with more data and epochs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report JSON"
    )

    args = parser.parse_args()

    report = run_evaluation(full=args.full, output_path=args.output)

    # Print final metrics table
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'Model':<12} {'Random':<12} {'Popularity':<12} {'Similarity':<12}")
    print("-" * 80)
    print(f"{'Precision@5':<25} {report.model_metrics.precision_at_5:<12.4f} "
          f"{report.baseline_random.precision_at_5:<12.4f} "
          f"{report.baseline_popularity.precision_at_5:<12.4f} "
          f"{report.baseline_similarity.precision_at_5:<12.4f}")
    print(f"{'Precision@10':<25} {report.model_metrics.precision_at_10:<12.4f} "
          f"{report.baseline_random.precision_at_10:<12.4f} "
          f"{report.baseline_popularity.precision_at_10:<12.4f} "
          f"{report.baseline_similarity.precision_at_10:<12.4f}")
    print(f"{'NDCG@10':<25} {report.model_metrics.ndcg_at_10:<12.4f} "
          f"{report.baseline_random.ndcg_at_10:<12.4f} "
          f"{report.baseline_popularity.ndcg_at_10:<12.4f} "
          f"{report.baseline_similarity.ndcg_at_10:<12.4f}")
    print(f"{'Recall@10':<25} {report.model_metrics.recall_at_10:<12.4f} "
          f"{report.baseline_random.recall_at_10:<12.4f} "
          f"{report.baseline_popularity.recall_at_10:<12.4f} "
          f"{report.baseline_similarity.recall_at_10:<12.4f}")
    print(f"{'Coverage':<25} {report.model_metrics.coverage:<12.2%} "
          f"{report.baseline_random.coverage:<12.2%} "
          f"{report.baseline_popularity.coverage:<12.2%} "
          f"{report.baseline_similarity.coverage:<12.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
