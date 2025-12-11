import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_ROOT = "/xdisk/cjgomez/joshdunlapc/word2vec_models"

DECADES = [
    "1850s", "1860s", "1870s", "1880s", "1890s",
    "1900s", "1910s", "1920s", "1930s"
]

TARGET_WORDS = ["utopia", "utopian"]

OUTPUT_ROOT = "analysis_outputs"
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
CSV_DIR = os.path.join(OUTPUT_ROOT, "csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Analogy test definitions
ANALOGY_TESTS = [
    {
        "name": "gender_brother_woman_minus_man",
        "positive": ["brother", "woman"],
        "negative": ["man"],
    },
    {
        "name": "geography_france_berlin_minus_paris",
        "positive": ["france", "berlin"],
        "negative": ["paris"],
    }
]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_model(decade):
    return Word2Vec.load(f"{MODEL_ROOT}/{decade}/word2vec_{decade}.model")


def get_similar_words_df(model, word, decade, topn=10):
    """Return dataframe of similar words."""
    if word not in model.wv:
        return pd.DataFrame([])

    similar_words = model.wv.most_similar(word, topn=topn)
    df = pd.DataFrame(similar_words, columns=["similar_word", "similarity"])
    df.insert(0, "word", word)
    df.insert(0, "decade", decade)
    return df


def run_analogy_test(model, test, decade):
    """Run an analogy (A is to B as C is to ?) and return a dataframe."""
    missing = [
        w for w in (test["positive"] + test["negative"])
        if w not in model.wv
    ]
    if missing:
        return pd.DataFrame([{
            "decade": decade,
            "test_name": test["name"],
            "status": "missing_words",
            "missing": ",".join(missing)
        }])

    try:
        result = model.wv.most_similar(
            positive=test["positive"],
            negative=test["negative"],
            topn=5
        )
        df = pd.DataFrame(result, columns=["predicted_word", "similarity"])
        df.insert(0, "decade", decade)
        df.insert(1, "test_name", test["name"])
        df["status"] = "ok"
        return df
    except Exception as e:
        return pd.DataFrame([{
            "decade": decade,
            "test_name": test["name"],
            "status": f"error: {str(e)}"
        }])


def plot_word_neighbors(word, model, decade):
    """TSNE visualization of word + nearest neighbors."""
    if word not in model.wv:
        return None

    neighbors = model.wv.most_similar(word, topn=15)
    words = [word] + [n for n, _ in neighbors]
    vectors = np.array([model.wv[w] for w in words])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    coords = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for (x, y), w in zip(coords, words):
        if w == word:
            plt.scatter(x, y, s=180, marker='*')
            plt.text(x + 0.01, y + 0.01, w, fontsize=14, weight='bold')
        else:
            plt.scatter(x, y, s=80)
            plt.text(x + 0.01, y + 0.01, w, fontsize=12)

    plt.title(f"{decade}: TSNE for '{word}' and neighbors", fontsize=16)
    outpath = f"{PLOTS_DIR}/{decade}_{word}_tsne.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath

# ---------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------
def main():
    all_similarities = []
    all_analogies = []

    print("\n====================================")
    print(" MULTI-DECADE WORD2VEC ANALYSIS")
    print("====================================")

    for decade in DECADES:
        print(f"\nRunning decade: {decade}")
        model = load_model(decade)

        # --- Similarity tests ---
        for word in TARGET_WORDS:
            df = get_similar_words_df(model, word, decade)
            if len(df) > 0:
                all_similarities.append(df)

            path = plot_word_neighbors(word, model, decade)
            if path:
                print(f"Saved plot: {path}")

        # --- Analogy solver tests ---
        for test in ANALOGY_TESTS:
            df = run_analogy_test(model, test, decade)
            all_analogies.append(df)

    # Save CSV outputs
    if all_similarities:
        combined = pd.concat(all_similarities, ignore_index=True)
        out_csv = f"{CSV_DIR}/similarities_utopia_across_decades.csv"
        combined.to_csv(out_csv, index=False)
        print(f"\n✓ Saved similarity CSV: {out_csv}")

    if all_analogies:
        combined = pd.concat(all_analogies, ignore_index=True)
        out_csv = f"{CSV_DIR}/analogy_tests_across_decades.csv"
        combined.to_csv(out_csv, index=False)
        print(f"✓ Saved analogy CSV: {out_csv}")

    print("\nAnalysis complete.\n")


if __name__ == "__main__":
    main()
