# src/features/model/jain_baseline.py
"""
Faithful re-implementation of Jain et al. (2022), "A content and URL analysis-based efficient
approach to detect smishing SMS in intelligent systems", Int J Intell Syst, doi:10.1002/int.23035.

Shared by notebooks/20_run_jain_replication.py (training + evaluation) and src/serving/predict.py
(Research Mode's live "Ensemble Baseline (Jain et al.)"), so the served model is exactly the one
the thesis tables measured.

Fig. 4 / Section 3: an SMS is scanned for a URL (UrlExtract, Section 4.2).
  NO URL  -> Text Phishing Classifier
    cleaning: labeling, removal of null + duplicate rows (Section 3.1)
    preprocessing (Section 3.2, Fig. 5): lowercase -> nltk tokenization -> removal of special
      characters -> removal of stopwords -> stemming
    features: TF-IDF, max_features=3000 (their best setting, Tables 2-4)
    balancing: SMOTE on the TF-IDF training matrix (the "_os" setting, their best)
    model: Voting classifier of KNN + Random Forest + Extra Trees (Section 3.4.3, Fig. 17)
  URL     -> URL Phishing Classifier
    cleaning: labeling, removal of null + duplicate URLs
    preprocessing (Section 3.2): RegexpTokenizer('[A-Za-z]+') -> Snowball stemmer
    features: CountVectorizer (Section 3.4.4b)
    model: Logistic Regression (Section 5.4 -- chosen over MNB/BNB for precision)

Needs NLTK's punkt, punkt_tab and stopwords data (python -m nltk.downloader punkt punkt_tab stopwords).
"""
import string

import numpy as np
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from urlextract import URLExtract

# Unavoidable deviations / details the paper leaves unspecified -- each one is the most literal
# reading available. Logged in every results JSON under 'replication_notes'.
REPLICATION_NOTES = [
    "URL classifier training data: the paper uses a separate 507,195-URL good/bad corpus; we train "
    "on the URLs extracted (UrlExtract) from our Training-split messages, labeled with their "
    "message's label, so both classifiers see only our Training split (no test leakage).",
    "URL 'removal of non-English entries' is skipped: our URLs are not language-tagged.",
    "Text stemmer: the paper names Snowball only for URLs; the text stemmer is unnamed, so NLTK's "
    "PorterStemmer (the NLTK default) is used.",
    "'Removal of special characters' keeps alphanumeric tokens only (str.isalnum), matching the "
    "paper's listed character set !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~.",
    "Hyperparameters are not reported in the paper: scikit-learn defaults throughout "
    "(random_state=seed); LogisticRegression max_iter raised to 1000 only so lbfgs converges.",
    "Voting type is not stated; 'predicts the output on the basis of voting' -> hard (majority) "
    "voting, scikit-learn's VotingClassifier default.",
    "SMOTE is applied to the text classifier only (Section 4.2 lists it there; the URL "
    "pipeline in Section 4.2 has none). It is fit on Training rows only, never on test rows.",
    "Messages with several URLs are routed on their first extracted URL.",
    "Duplicate removal applies to the Training split only; the Test split is left untouched so "
    "every model is scored on identical rows.",
]
TFIDF_MAX_FEATURES = 3000

# Module-level, not attributes of the detector, so a saved detector pickles cleanly.
_STOPWORDS = set(stopwords.words('english'))
_PUNCT = set(string.punctuation)
_porter = PorterStemmer()
_snowball = SnowballStemmer('english')
_url_tokenizer = RegexpTokenizer(r'[A-Za-z]+')
_url_extractor = URLExtract()


def transform_text(text: str) -> str:
    tokens = word_tokenize(text.lower())                                    # lowercase + tokenization
    tokens = [t for t in tokens if t.isalnum()]                              # removal of special characters
    tokens = [t for t in tokens if t not in _STOPWORDS and t not in _PUNCT]  # removal of stopwords
    return ' '.join(_porter.stem(t) for t in tokens)                         # stemming


def transform_url(url: str) -> str:
    return ' '.join(_snowball.stem(t) for t in _url_tokenizer.tokenize(url))  # tokenization + stemming


def find_urls(text: str) -> list[str]:
    return _url_extractor.find_urls(text)


def first_url(text: str):
    urls = find_urls(text)
    return urls[0] if urls else None


class JainSmishingDetector:
    """Both classifiers plus the URL-presence router. Labels are 0 = legitimate, 1 = smishing."""

    def __init__(self, seed: int):
        self.seed = seed

    def use_single_core_inference(self) -> 'JainSmishingDetector':
        # Training fits the forests with n_jobs=-1 for speed. For one message at a time that only
        # adds thread-pool overhead, so serving and latency timing predict single-core.
        # Predictions are identical either way (n_jobs never changes a fitted tree).
        for est in self.text_clf.estimators_:
            if hasattr(est, 'n_jobs'):
                est.n_jobs = 1
        return self

    def learned_value_counts(self) -> dict:
        # This model has no neural "trainable parameters", so its size is reported as the number
        # of numeric values it learned from the training data and must store to predict:
        #   TF-IDF idf weights; KNN's stored (SMOTE-balanced) training vectors (non-zeros) + their
        #   labels; one split threshold per internal tree node and n_classes values per leaf, over
        #   every tree in the Random Forest and Extra Trees; Logistic Regression's weights + bias.
        def forest_values(forest):
            total = 0
            for tree in forest.estimators_:
                t = tree.tree_
                leaves = int((t.children_left == -1).sum())
                total += int(t.node_count - leaves) + leaves * int(t.n_classes[0])
            return total
        knn, rf, etc = (self.text_clf.named_estimators_[k] for k in ('knn', 'rf', 'etc'))
        counts = {
            'tfidf_idf_weights': int(self.tfidf.idf_.size),
            'knn_stored_training_values': int(knn._fit_X.nnz + knn._y.size),
            'random_forest_tree_values': forest_values(rf),
            'extra_trees_tree_values': forest_values(etc),
            'url_logistic_regression_weights': int(self.url_clf.coef_.size + self.url_clf.intercept_.size),
        }
        counts['total'] = sum(counts.values())
        return counts

    def fit(self, texts: list[str], labels) -> 'JainSmishingDetector':
        labels = np.asarray(labels)
        self.text_info = self._fit_text_classifier(texts, labels)
        self.url_info = self._fit_url_classifier(texts, labels)
        return self

    def _fit_text_classifier(self, texts, labels) -> dict:
        # removal of null + duplicate rows (first occurrence kept)
        seen, keep = set(), []
        for i, t in enumerate(texts):
            if isinstance(t, str) and t not in seen:
                seen.add(t)
                keep.append(i)
        y = labels[keep]
        self.tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        X = self.tfidf.fit_transform([transform_text(texts[i]) for i in keep])
        X_os, y_os = SMOTE(random_state=self.seed).fit_resample(X, y)
        self.text_clf = VotingClassifier([
            ('knn', KNeighborsClassifier()),
            ('rf', RandomForestClassifier(random_state=self.seed, n_jobs=-1)),
            ('etc', ExtraTreesClassifier(random_state=self.seed, n_jobs=-1)),
        ], voting='hard')
        self.text_clf.fit(X_os, y_os)
        return {'n_rows_after_cleaning': len(keep), 'n_duplicates_removed': len(texts) - len(keep),
                'class_counts_before_smote': np.bincount(y, minlength=2).tolist(),
                'class_counts_after_smote': np.bincount(y_os, minlength=2).tolist(),
                'tfidf_vocabulary_size': len(self.tfidf.vocabulary_)}

    def _fit_url_classifier(self, texts, labels) -> dict:
        rows = []
        for text, label in zip(texts, labels):
            if isinstance(text, str):
                rows += [(u, int(label)) for u in find_urls(text)]
        unique = list(dict.fromkeys(rows))  # removal of duplicate (url, label) pairs, order kept
        urls, y = [u for u, _ in unique], np.array([l for _, l in unique])
        self.url_vec = CountVectorizer()
        X = self.url_vec.fit_transform([transform_url(u) for u in urls])
        self.url_clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        self.url_clf.fit(X, y)
        return {'n_urls_extracted': len(rows), 'n_unique_urls': len(unique),
                'class_counts': np.bincount(y, minlength=2).tolist(),
                'count_vocabulary_size': len(self.url_vec.vocabulary_)}

    def predict_text_classifier(self, texts: list[str]) -> np.ndarray:
        return self.text_clf.predict(self.tfidf.transform([transform_text(t) for t in texts]))

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        # Returns (predictions, routed) -- routed[i] is True when message i went to the URL classifier.
        preds = self.predict_text_classifier(texts)
        urls = [first_url(t) for t in texts]
        routed = np.array([u is not None for u in urls])
        if routed.any():
            routed_urls = [transform_url(u) for u in urls if u is not None]
            preds[routed] = self.url_clf.predict(self.url_vec.transform(routed_urls))
        return preds, routed

    def predict_one(self, text: str) -> dict:
        # Serving path. prob_smishing is the routed classifier's own score: Logistic Regression's
        # probability on the URL route; on the text route, the share of the three hard voters
        # (KNN / RF / Extra Trees) that vote smishing -- 0, 1/3, 2/3 or 1. The label is always
        # exactly what predict() returns (majority vote / LR argmax).
        url = first_url(text)
        if url is not None:
            prob = float(self.url_clf.predict_proba(self.url_vec.transform([transform_url(url)]))[0, 1])
            label = int(self.url_clf.predict(self.url_vec.transform([transform_url(url)]))[0])
            return {'label': label, 'prob_smishing': prob, 'route': 'url_classifier', 'routed_url': url}
        X = self.tfidf.transform([transform_text(text)])
        votes = [int(est.predict(X)[0]) for est in self.text_clf.estimators_]
        label = int(self.text_clf.predict(X)[0])
        return {'label': label, 'prob_smishing': sum(votes) / len(votes),
                'route': 'text_classifier', 'routed_url': None}
