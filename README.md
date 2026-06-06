# Document Clustering & Topic Modelling
### 20 Newsgroups — 5 Categories | TF-IDF · K-Means · LDA · PCA · t-SNE

Unsupervised clustering and topic discovery on real-world newsgroup posts — no labels used during training. Evaluated with silhouette score, cluster purity, NMI, and ARI.

---

## Overview

**Can a machine group documents by topic with zero labels?**

Pipeline:
1. Load 5 categories from the 20 Newsgroups dataset
2. Clean and preprocess raw posts (strip boilerplate, stopwords, lemmatize)
3. Represent as TF-IDF vectors (8,000 features, bigrams)
4. Cluster with **K-Means** (k=5, k-means++ init)
5. Discover topics with **LDA** (Latent Dirichlet Allocation)
6. Visualize in 2D with **PCA** (TruncatedSVD) and **t-SNE**
7. Evaluate with silhouette, purity, NMI, ARI

---

## Dataset

**20 Newsgroups** — fetched automatically via `sklearn.datasets.fetch_20newsgroups`

| Category | Topic |
|----------|-------|
| `sci.med` | Medical science |
| `sci.space` | Space & astronomy |
| `comp.graphics` | Computer graphics |
| `rec.sport.hockey` | Hockey |
| `talk.politics.guns` | Gun politics |

Headers, footers, and quoted replies are stripped. Documents under 20 words are dropped. Final corpus: ~4,300+ documents.

---

## Preprocessing

Raw newsgroup posts are noisy. The pipeline:

- Lowercase → strip URLs, emails, file paths
- Remove non-alphabetic characters
- Drop English stopwords + domain-specific noise words (`nntp`, `writes`, `article`, `posting`, `edu`, `university`, etc.)
- Keep tokens with 3–25 characters only
- **Lemmatize** with WordNetLemmatizer
- Drop tokens with only one unique character (e.g. `"aaaa"`)

---

## TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features = 8000,
    ngram_range  = (1, 2),   # unigrams + bigrams
    min_df       = 5,
    max_df       = 0.85,
    sublinear_tf = True      # log(1 + tf) dampening
)
```

Matrix: ~4,300 docs × 8,000 features | ~99.2% sparse

---

## K-Means Clustering

```python
KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300)
```

k=5 selected via the elbow method (inertia curve) and confirmed by the known category count.

---

## LDA Topic Modelling

```python
LatentDirichletAllocation(
    n_components    = 5,
    doc_topic_prior = 0.1,    # sparse document-topic prior (α)
    topic_word_prior= 0.01,   # sparse topic-word prior (β)
    learning_method = 'online',
    max_iter        = 30
)
```

Trained on raw counts (CountVectorizer) — LDA requires count input, not TF-IDF.

4 of 5 discovered topics mapped cleanly to known categories:

| Topic | Top Words | Matched Category |
|-------|-----------|-----------------|
| 1 | image, file, jpeg, color, program… | comp.graphics |
| 2 | space, nasa, orbit, launch, earth… | sci.space |
| 3 | game, team, play, season, player… | rec.sport.hockey |
| 4 | gun, weapon, law, firearm, right… | talk.politics.guns |
| 5 | people, time, make, right, thing… | Mixed (cross-category) |

---

## Results

| Metric | Score | Notes |
|--------|-------|-------|
| Silhouette Score (cosine) | ~0.018 | Low is normal for high-dim sparse text |
| Cluster Purity | ~0.669 | Fraction of each cluster matching majority class |
| NMI | ~0.565 | 0 = random, 1 = perfect |

> **On silhouette score:** Values below 0.20 are completely expected for 8,000-feature sparse TF-IDF matrices. In high-dimensional space, most documents appear equidistant. **NMI and purity are better indicators here.**

---

## What's in the Notebook

- Document length distribution + category balance (EDA)
- Elbow curve for choosing k
- Top-10 TF-IDF terms per K-Means cluster
- LDA top-8 words per topic (bar charts)
- PCA 2D scatter: K-Means clusters vs true categories (side by side)
- t-SNE 2D scatter: K-Means clusters vs true categories (side by side)
- Cluster × category contingency heatmap
- Per-cluster purity, NMI, ARI summary

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

NLTK data (`stopwords`, `wordnet`, `omw-1.4`) downloads automatically in the notebook.

> **t-SNE note:** PCA to 50D is applied first for speed, then t-SNE to 2D. Expect 2–5 min on CPU.

---

## Key Takeaways

- **TF-IDF + K-Means** achieves ~67% purity with zero labels — a strong unsupervised baseline.
- **Don't over-interpret silhouette score for text** — it's structurally low in high-dimensional sparse spaces. NMI and purity tell the real story.
- **LDA found 4 clean topics** out of 5 with no supervision — the 5th absorbed generic cross-category posts, which is typical LDA behavior.
- **t-SNE reveals cluster structure** that PCA misses — the 5 categories form visually distinct regions in t-SNE space, confirming the TF-IDF representations carry real topic signal.
