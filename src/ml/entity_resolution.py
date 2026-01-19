"""
Entity Resolution for Multi-CRM Account Matching

Identifies when accounts across different Salesforce orgs represent the same
real-world company. Uses a multi-stage approach:

1. Blocking: Reduce comparison space using domain/name prefixes
2. Deterministic matching: Exact matches on normalized identifiers
3. Fuzzy matching: Token-based and edit-distance similarity
4. ML scoring: Trained classifier for ambiguous cases

Evaluation metrics: Precision, Recall, F1 on labeled match pairs.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of entity matching between two accounts"""
    account1_id: str
    account2_id: str
    match_score: float
    match_type: str  # 'exact', 'fuzzy', 'ml'
    confidence: str  # 'high', 'medium', 'low'
    match_reasons: List[str]


@dataclass
class EntityResolutionMetrics:
    """Evaluation metrics for entity resolution"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    threshold: float


class NameNormalizer:
    """Normalize company names for comparison"""

    # Common suffixes to remove
    SUFFIXES = {
        'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
        'llc', 'llp', 'ltd', 'limited', 'plc', 'gmbh', 'ag', 'sa', 'nv',
        'holdings', 'group', 'international', 'intl', 'global', 'worldwide'
    }

    # Common abbreviations to expand
    ABBREVS = {
        'intl': 'international',
        'corp': 'corporation',
        'inc': 'incorporated',
        'co': 'company',
        'tech': 'technology',
        'sys': 'systems',
        'svc': 'services',
        'svcs': 'services',
        'mfg': 'manufacturing',
    }

    @classmethod
    def normalize(cls, name: str) -> str:
        """Normalize a company name for matching"""
        if not name:
            return ""

        # Lowercase
        name = name.lower().strip()

        # Remove punctuation except ampersand
        name = re.sub(r'[^\w\s&]', ' ', name)

        # Expand abbreviations
        tokens = name.split()
        tokens = [cls.ABBREVS.get(t, t) for t in tokens]

        # Remove common suffixes
        tokens = [t for t in tokens if t not in cls.SUFFIXES]

        # Remove extra whitespace
        name = ' '.join(tokens)
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    @classmethod
    def extract_tokens(cls, name: str) -> Set[str]:
        """Extract significant tokens from a name"""
        normalized = cls.normalize(name)
        tokens = set(normalized.split())
        # Remove very short tokens
        return {t for t in tokens if len(t) > 2}


class DomainNormalizer:
    """Normalize and extract domains from URLs/emails"""

    @classmethod
    def extract_domain(cls, url_or_email: str) -> Optional[str]:
        """Extract normalized domain from URL or email"""
        if not url_or_email:
            return None

        url_or_email = url_or_email.lower().strip()

        # Handle email
        if '@' in url_or_email:
            domain = url_or_email.split('@')[-1]
        else:
            # Handle URL
            if not url_or_email.startswith(('http://', 'https://')):
                url_or_email = 'https://' + url_or_email
            try:
                parsed = urlparse(url_or_email)
                domain = parsed.netloc or parsed.path.split('/')[0]
            except Exception:
                domain = url_or_email

        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]

        # Remove trailing slashes/paths
        domain = domain.split('/')[0]

        return domain if domain else None

    @classmethod
    def get_root_domain(cls, domain: str) -> Optional[str]:
        """Get root domain (e.g., 'google.com' from 'mail.google.com')"""
        if not domain:
            return None

        parts = domain.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return domain


class SimilarityMetrics:
    """Text similarity metrics for fuzzy matching"""

    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Levenshtein (edit) distance between two strings"""
        if len(s1) < len(s2):
            return SimilarityMetrics.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def normalized_levenshtein(s1: str, s2: str) -> float:
        """Normalized Levenshtein similarity (0-1 scale)"""
        if not s1 or not s2:
            return 0.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = SimilarityMetrics.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    @staticmethod
    def jaro_winkler(s1: str, s2: str) -> float:
        """Jaro-Winkler similarity (favors strings with common prefixes)"""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 +
                (matches - transpositions / 2) / matches) / 3

        # Winkler modification: boost for common prefix
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)


class EntityResolver:
    """
    Main entity resolution engine for matching accounts across CRM orgs.

    Usage:
        resolver = EntityResolver()
        matches = resolver.find_matches(accounts_org1, accounts_org2)

        # With ML model
        resolver.train(labeled_pairs)
        matches = resolver.find_matches(accounts_org1, accounts_org2, use_ml=True)
    """

    def __init__(
        self,
        domain_weight: float = 0.4,
        name_weight: float = 0.4,
        metadata_weight: float = 0.2,
        match_threshold: float = 0.7
    ):
        self.domain_weight = domain_weight
        self.name_weight = name_weight
        self.metadata_weight = metadata_weight
        self.match_threshold = match_threshold

        self.name_normalizer = NameNormalizer()
        self.domain_normalizer = DomainNormalizer()
        self.similarity = SimilarityMetrics()

        self.ml_model: Optional[RandomForestClassifier] = None
        self.is_trained = False

    def _create_blocking_key(self, account: pd.Series) -> str:
        """Create blocking key to reduce comparison space"""
        # Use first 3 chars of normalized name + country
        name = self.name_normalizer.normalize(account.get('Name', ''))
        country = str(account.get('BillingCountry', '')).lower()[:3]

        prefix = name[:3] if len(name) >= 3 else name
        return f"{prefix}_{country}"

    def _extract_features(
        self,
        account1: pd.Series,
        account2: pd.Series
    ) -> Dict[str, float]:
        """Extract features for a pair of accounts"""
        features = {}

        # Domain matching
        domain1 = self.domain_normalizer.extract_domain(account1.get('Website', ''))
        domain2 = self.domain_normalizer.extract_domain(account2.get('Website', ''))

        if domain1 and domain2:
            root1 = self.domain_normalizer.get_root_domain(domain1)
            root2 = self.domain_normalizer.get_root_domain(domain2)
            features['domain_exact_match'] = 1.0 if root1 == root2 else 0.0
            features['domain_similarity'] = self.similarity.jaro_winkler(
                domain1 or '', domain2 or ''
            )
        else:
            features['domain_exact_match'] = 0.0
            features['domain_similarity'] = 0.0

        # Name matching
        name1 = self.name_normalizer.normalize(account1.get('Name', ''))
        name2 = self.name_normalizer.normalize(account2.get('Name', ''))

        features['name_exact_match'] = 1.0 if name1 == name2 else 0.0
        features['name_jaro_winkler'] = self.similarity.jaro_winkler(name1, name2)
        features['name_levenshtein'] = self.similarity.normalized_levenshtein(name1, name2)

        tokens1 = self.name_normalizer.extract_tokens(account1.get('Name', ''))
        tokens2 = self.name_normalizer.extract_tokens(account2.get('Name', ''))
        features['name_jaccard'] = self.similarity.jaccard_similarity(tokens1, tokens2)

        # Metadata matching
        features['same_country'] = 1.0 if (
            account1.get('BillingCountry') == account2.get('BillingCountry')
            and account1.get('BillingCountry')
        ) else 0.0

        features['same_industry'] = 1.0 if (
            account1.get('Industry') == account2.get('Industry')
            and account1.get('Industry')
        ) else 0.0

        # Size similarity
        emp1 = account1.get('NumberOfEmployees', 0) or 0
        emp2 = account2.get('NumberOfEmployees', 0) or 0
        if emp1 > 0 and emp2 > 0:
            features['size_ratio'] = min(emp1, emp2) / max(emp1, emp2)
        else:
            features['size_ratio'] = 0.5  # Unknown

        return features

    def _compute_score(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """Compute match score from features"""
        reasons = []

        # Domain score
        if features['domain_exact_match'] > 0:
            domain_score = 1.0
            reasons.append("Exact domain match")
        else:
            domain_score = features['domain_similarity']
            if domain_score > 0.8:
                reasons.append(f"Similar domain ({domain_score:.2f})")

        # Name score
        if features['name_exact_match'] > 0:
            name_score = 1.0
            reasons.append("Exact name match")
        else:
            name_score = max(
                features['name_jaro_winkler'],
                features['name_jaccard'],
                features['name_levenshtein']
            )
            if name_score > 0.8:
                reasons.append(f"Similar name ({name_score:.2f})")

        # Metadata score
        metadata_score = (
            features['same_country'] * 0.4 +
            features['same_industry'] * 0.3 +
            features['size_ratio'] * 0.3
        )
        if features['same_country'] > 0:
            reasons.append("Same country")
        if features['same_industry'] > 0:
            reasons.append("Same industry")

        # Weighted combination
        total_score = (
            self.domain_weight * domain_score +
            self.name_weight * name_score +
            self.metadata_weight * metadata_score
        )

        return total_score, reasons

    def match_pair(
        self,
        account1: pd.Series,
        account2: pd.Series,
        use_ml: bool = False
    ) -> Optional[MatchResult]:
        """Check if two accounts represent the same entity"""
        features = self._extract_features(account1, account2)

        # Exact match shortcut
        if features['domain_exact_match'] > 0 and features['name_exact_match'] > 0:
            return MatchResult(
                account1_id=str(account1.get('Id', '')),
                account2_id=str(account2.get('Id', '')),
                match_score=1.0,
                match_type='exact',
                confidence='high',
                match_reasons=['Exact domain and name match']
            )

        # ML prediction if available
        if use_ml and self.is_trained:
            feature_vector = np.array([list(features.values())])
            ml_score = self.ml_model.predict_proba(feature_vector)[0, 1]
            if ml_score >= self.match_threshold:
                return MatchResult(
                    account1_id=str(account1.get('Id', '')),
                    account2_id=str(account2.get('Id', '')),
                    match_score=float(ml_score),
                    match_type='ml',
                    confidence='high' if ml_score > 0.9 else 'medium',
                    match_reasons=[f'ML classifier score: {ml_score:.2f}']
                )

        # Rule-based scoring
        score, reasons = self._compute_score(features)

        if score >= self.match_threshold:
            confidence = 'high' if score > 0.9 else ('medium' if score > 0.8 else 'low')
            return MatchResult(
                account1_id=str(account1.get('Id', '')),
                account2_id=str(account2.get('Id', '')),
                match_score=score,
                match_type='fuzzy',
                confidence=confidence,
                match_reasons=reasons
            )

        return None

    def find_matches(
        self,
        accounts1: pd.DataFrame,
        accounts2: pd.DataFrame,
        use_ml: bool = False,
        use_blocking: bool = True
    ) -> List[MatchResult]:
        """Find all matching accounts between two orgs"""
        matches = []

        # Create blocking keys if enabled
        if use_blocking:
            accounts1 = accounts1.copy()
            accounts2 = accounts2.copy()
            accounts1['_block_key'] = accounts1.apply(self._create_blocking_key, axis=1)
            accounts2['_block_key'] = accounts2.apply(self._create_blocking_key, axis=1)

            # Group by blocking key
            blocks1 = accounts1.groupby('_block_key')
            blocks2_dict = {k: g for k, g in accounts2.groupby('_block_key')}

            for block_key, group1 in blocks1:
                if block_key not in blocks2_dict:
                    continue
                group2 = blocks2_dict[block_key]

                for _, acc1 in group1.iterrows():
                    for _, acc2 in group2.iterrows():
                        match = self.match_pair(acc1, acc2, use_ml)
                        if match:
                            matches.append(match)
        else:
            # Full comparison (O(n*m))
            for _, acc1 in accounts1.iterrows():
                for _, acc2 in accounts2.iterrows():
                    match = self.match_pair(acc1, acc2, use_ml)
                    if match:
                        matches.append(match)

        # Sort by score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        logger.info(f"Found {len(matches)} matches between {len(accounts1)} and {len(accounts2)} accounts")
        return matches

    def train(
        self,
        labeled_pairs: pd.DataFrame,
        accounts: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Train ML model on labeled match pairs.

        Args:
            labeled_pairs: DataFrame with columns [account1_id, account2_id, is_match]
            accounts: DataFrame of all accounts

        Returns:
            Training metrics
        """
        X = []
        y = []

        accounts_dict = {str(row['Id']): row for _, row in accounts.iterrows()}

        for _, pair in labeled_pairs.iterrows():
            acc1_id = str(pair['account1_id'])
            acc2_id = str(pair['account2_id'])

            if acc1_id not in accounts_dict or acc2_id not in accounts_dict:
                continue

            acc1 = accounts_dict[acc1_id]
            acc2 = accounts_dict[acc2_id]

            features = self._extract_features(acc1, acc2)
            X.append(list(features.values()))
            y.append(pair['is_match'])

        X = np.array(X)
        y = np.array(y)

        # Train Random Forest
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )

        # Cross-validation
        cv_scores = cross_val_score(self.ml_model, X, y, cv=5, scoring='f1')

        # Fit final model
        self.ml_model.fit(X, y)
        self.is_trained = True

        logger.info(f"Trained entity resolution model. CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        return {
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'training_samples': len(X),
            'positive_rate': float(y.mean())
        }

    def evaluate(
        self,
        test_pairs: pd.DataFrame,
        accounts: pd.DataFrame,
        threshold: float = None
    ) -> EntityResolutionMetrics:
        """
        Evaluate entity resolution on labeled test set.

        Args:
            test_pairs: DataFrame with [account1_id, account2_id, is_match]
            accounts: DataFrame of all accounts
            threshold: Score threshold (uses self.match_threshold if None)

        Returns:
            EntityResolutionMetrics
        """
        if threshold is None:
            threshold = self.match_threshold

        accounts_dict = {str(row['Id']): row for _, row in accounts.iterrows()}

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for _, pair in test_pairs.iterrows():
            acc1_id = str(pair['account1_id'])
            acc2_id = str(pair['account2_id'])
            is_match = pair['is_match']

            if acc1_id not in accounts_dict or acc2_id not in accounts_dict:
                continue

            acc1 = accounts_dict[acc1_id]
            acc2 = accounts_dict[acc2_id]

            result = self.match_pair(acc1, acc2, use_ml=self.is_trained)
            predicted_match = result is not None and result.match_score >= threshold

            if predicted_match and is_match:
                true_positives += 1
            elif predicted_match and not is_match:
                false_positives += 1
            elif not predicted_match and is_match:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return EntityResolutionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            threshold=threshold
        )


def generate_synthetic_test_data(
    n_accounts: int = 200,
    match_rate: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data for entity resolution testing.

    Returns:
        accounts1: Accounts from "org1"
        accounts2: Accounts from "org2" (some are same entities as org1)
        labeled_pairs: Ground truth labels
    """
    np.random.seed(seed)

    base_companies = [
        ("Acme Corporation", "acme.com", "Technology", "USA"),
        ("Global Finance Inc", "globalfinance.com", "Finance", "UK"),
        ("HealthCare Plus", "healthcareplus.org", "Healthcare", "USA"),
        ("TechStart Systems", "techstart.io", "Technology", "Germany"),
        ("Retail Giants Ltd", "retailgiants.co.uk", "Retail", "UK"),
    ]

    def create_variant(name: str, domain: str) -> Tuple[str, str]:
        """Create a realistic variant of company name/domain"""
        variations = [
            (name, domain),  # Exact
            (name + " Inc", domain),  # Suffix added
            (name.replace("Corporation", "Corp"), domain),  # Abbreviated
            (name, "www." + domain),  # www prefix
            (name.upper(), domain),  # Case change
            (name.replace(" ", "-"), domain),  # Hyphenated
        ]
        return variations[np.random.randint(len(variations))]

    accounts1 = []
    accounts2 = []
    labeled_pairs = []

    for i in range(n_accounts):
        # Create account for org1
        base = base_companies[i % len(base_companies)]
        acc1 = {
            'Id': f'ORG1_ACC_{i:04d}',
            'Name': base[0] + f" #{i}",
            'Website': f"company{i}.{base[1].split('.')[-1]}",
            'Industry': base[2],
            'BillingCountry': base[3],
            'NumberOfEmployees': np.random.randint(10, 5000)
        }
        accounts1.append(acc1)

        # Decide if this account should have a match in org2
        if np.random.random() < match_rate:
            # Create matching account with variations
            var_name, var_domain = create_variant(acc1['Name'], acc1['Website'])
            acc2 = {
                'Id': f'ORG2_ACC_{i:04d}',
                'Name': var_name,
                'Website': var_domain,
                'Industry': acc1['Industry'],
                'BillingCountry': acc1['BillingCountry'],
                'NumberOfEmployees': acc1['NumberOfEmployees'] + np.random.randint(-100, 100)
            }
            labeled_pairs.append({
                'account1_id': acc1['Id'],
                'account2_id': acc2['Id'],
                'is_match': 1
            })
        else:
            # Create non-matching account
            other_base = base_companies[(i + 1) % len(base_companies)]
            acc2 = {
                'Id': f'ORG2_ACC_{i:04d}',
                'Name': other_base[0] + f" #{i + 100}",
                'Website': f"other{i}.{other_base[1].split('.')[-1]}",
                'Industry': other_base[2],
                'BillingCountry': other_base[3],
                'NumberOfEmployees': np.random.randint(10, 5000)
            }
            labeled_pairs.append({
                'account1_id': acc1['Id'],
                'account2_id': acc2['Id'],
                'is_match': 0
            })

        accounts2.append(acc2)

    return (
        pd.DataFrame(accounts1),
        pd.DataFrame(accounts2),
        pd.DataFrame(labeled_pairs)
    )
