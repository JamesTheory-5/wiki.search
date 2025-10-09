# wiki.search

```python
from datasets import load_dataset
import pickle, os, sys

wiki = load_dataset("wikipedia", "20220301.en", split="train")
cache_path = "titles.cache"

# Build or load title index
if os.path.exists(cache_path):
    title_index = pickle.load(open(cache_path, "rb"))
else:
    title_index = {t.lower(): i for i, t in enumerate(wiki["title"])}
    pickle.dump(title_index, open(cache_path, "wb"))

# Fast lookup
def search_cached(keyword: str, k: int = 5):
    keyword = keyword.lower()
    results = [(t, i) for t, i in title_index.items() if keyword in t][:k]
    return results

query = sys.argv[1]
for title, idx in search_cached(query.lower()):
    print(title)
    print(wiki[idx]["text"][:300], "\n")

```

### WIKI.SEARCH.INDEX.2

```python
from datasets import load_dataset
from transformers import pipeline
import pickle, os, sys

# Load subset of Wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split="train")
cache_path = "titles.cache"

# Cache or load title index
if os.path.exists(cache_path):
    title_index = pickle.load(open(cache_path, "rb"))
else:
    title_index = {t.lower(): i for i, t in enumerate(wiki["title"])}
    pickle.dump(title_index, open(cache_path, "wb"))

# NER pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")

def extract_entity(query: str) -> str:
    ents = [e["word"] for e in ner(query) if e["entity"].startswith("B-")]
    return " ".join(ents) if ents else None

def search_cached(keyword: str, k: int = 5):
    keyword = keyword.lower()
    return [(t, i) for t, i in title_index.items() if keyword in t][:k]

def answer_query(query: str):
    entity = extract_entity(query)
    if not entity:
        return "No entity found."
    matches = search_cached(entity)
    if not matches:
        return f"No articles found for '{entity}'."
    return matches

# Example
q = str(sys.argv[1])
hits = answer_query(q)

if isinstance(hits, str):
    print(hits)
else:
    for title, idx in hits:
        print(f"Title: {title}")
        print(f"Snippet: {wiki[idx]['text'][:300]}\n")

```

### WIKI.SEARCH.INDEX.3

```python
from datasets import load_dataset
from transformers import pipeline
import pickle, os, sys

# Load subset of Wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split="train")
cache_path = "titles.cache"
# Cache or load title index
if os.path.exists(cache_path):
    title_index = pickle.load(open(cache_path, "rb"))
else:
    title_index = {t.lower(): i for i, t in enumerate(wiki["title"])}
    pickle.dump(title_index, open(cache_path, "wb"))

def search_cached(keyword: str, k: int = 5):
    keyword = keyword.lower()
    return [(t, i) for t, i in title_index.items() if keyword in t][:k]

# POS tagging pipeline
pos_tagger = pipeline(
    "token-classification",
    model="vblagoje/bert-english-uncased-finetuned-pos",
    aggregation_strategy="simple"
)

def extract_noun_phrases(query: str):
    tokens = pos_tagger(query)
    return [tok["word"] for tok in tokens if tok["entity_group"] in ["NOUN", "PROPN"]]

def answer_query(query: str, k: int = 5):
    noun_chunks = extract_noun_phrases(query)
    results = []
    for chunk in noun_chunks:
        hits = search_cached(chunk, k=k)
        results.extend(hits)
    return results

# Example
q = sys.argv[1]
hits = answer_query(q, k=3)

for title, idx in hits:
    print(f"Title: {title}")
    print(f"Snippet: {wiki[idx]['text'][:300]}\n")
```

### WIKI.SEARCH.INDEX.4

```python
from datasets import load_dataset
from transformers import pipeline
from rapidfuzz import process, fuzz
import pickle, os, sys

# Load subset of Wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split="train")
cache_path = "titles.cache"

# Cache or load title index
if os.path.exists(cache_path):
    title_index = pickle.load(open(cache_path, "rb"))
else:
    title_index = {t.lower(): i for i, t in enumerate(wiki["title"])}
    pickle.dump(title_index, open(cache_path, "wb"))

titles = list(title_index.keys())  # list of lowercase titles for fuzzy matching

# Named-entity recognizer
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def extract_entities(query: str):
    tokens = ner(query)
    return [tok["word"].replace("##", "").strip() for tok in tokens]

def search_cached(keyword: str, k: int = 5):
    keyword = keyword.lower()
    hits = [(t, i) for t, i in title_index.items() if keyword in t][:k]
    return [(title, wiki[idx]["text"], 100) for title, idx in hits]

def fuzzy_lookup(ent: str, k: int = 3):
    matches = process.extract(ent.lower(), titles, scorer=fuzz.WRatio, limit=k)
    results = []
    for match_title, score, _ in matches:
        idx = title_index[match_title]
        results.append((match_title, wiki[idx]["text"], score))
    return results

def answer_query(query: str, k: int = 3):
    ents = extract_entities(query)
    if not ents:
        return []
    for ent in ents:
        direct = search_cached(ent, k)
        if direct:
            return direct
        fuzzy = fuzzy_lookup(ent, k)
        if fuzzy:
            return fuzzy
    return []

# Example
q = sys.argv[1] if len(sys.argv) > 1 else "Mechanic"
hits = answer_query(q, k=3)

for title, text, score in hits:
    print(f"{title} (score={score})")
    print(text[:300], "\n")
```

### WIKI.SEARCH.INDEX.5

```python
from datasets import load_dataset
from transformers import pipeline
from rapidfuzz import process, fuzz
import pickle, os, sys

wiki = load_dataset("wikipedia", "20220301.en", split="train[:100000]")
cache_path = "titles.cache"

if os.path.exists(cache_path):
    title_index = pickle.load(open(cache_path, "rb"))
else:
    title_index = {t.lower(): i for i, t in enumerate(wiki["title"])}
    pickle.dump(title_index, open(cache_path, "wb"))

titles = list(title_index.keys())
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def extract_entities(query):
    return [t["word"].replace("##", "").strip() for t in ner(query)]

def search_cached(ent, k=5):
    ent = ent.lower()
    hits = [(t, i) for t, i in title_index.items() if ent in t][:k]
    return [(wiki[i]["title"], wiki[i]["text"], 100) for _, i in hits]

def fuzzy_lookup(ent, k=3):
    matches = process.extract(ent.lower(), titles, scorer=fuzz.WRatio, limit=k)
    out = []
    for title_lc, score, _ in matches:
        idx = title_index[title_lc]
        out.append((wiki[idx]["title"], wiki[idx]["text"], score))
    return out

def answer_query(query, k=3):
    ents = extract_entities(query)
    if not ents:
        ents = [query]       # fallback
    results = []
    for ent in ents:
        results.extend(search_cached(ent, k))
        if not results:
            results.extend(fuzzy_lookup(ent, k))
    return results[:k]

q = sys.argv[1] if len(sys.argv) > 1 else "Mechanic"
for title, text, score in answer_query(q):
    print(f"{title} (score={score})\n{text[:200]}\n")

```


### WIKI.SEARCH.INDEX.6

```python
from datasets import load_dataset
from transformers import pipeline
from rapidfuzz import process, fuzz
import os, pickle, re

# smaller subset for RAM safety
wiki = load_dataset("wikipedia", "20220301.en", split="train[:200000]")
titles = wiki["title"]

# build or load cache
if os.path.exists("title_map"):
    with open("title_map", "rb") as f:
        title_map = pickle.load(f)
else:
    title_map = {t.lower(): i for i, t in enumerate(titles)}
    with open("title_map", "wb") as f:
        pickle.dump(title_map, f)

# pipelines
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
pos_tagger = pipeline(
    "token-classification",
    model="vblagoje/bert-english-uncased-finetuned-pos",
    aggregation_strategy="simple"
)

def extract_candidates(query):
    ents = [tok["word"] for tok in ner(query)]
    if ents:
        return ents
    tokens = pos_tagger(query)
    phrases, current = [], []
    for tok in tokens:
        if tok["entity_group"] in ["NOUN", "PROPN", "ADJ"]:
            current.append(tok["word"])
        elif tok["word"].lower() in ["of", "in", "on", "for"] and current:
            current.append(tok["word"])
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))
    return phrases or [query]

def search_cached(ent, k=5):
    ent = ent.lower()
    hits = [(t, i) for t, i in title_map.items() if ent in t][:k]
    return [(wiki[i]["title"], wiki[i]["text"], 100) for _, i in hits]

def fuzzy_lookup(ent, k=3, min_score=70):
    matches = process.extract(ent.lower(), list(title_map.keys()), scorer=fuzz.WRatio, limit=k)
    results = []
    for match_title, score, _ in matches:
        if score >= min_score:
            idx = title_map[match_title]
            results.append((wiki[idx]["title"], wiki[idx]["text"], score))
    return results

def is_disambiguation(text):
    s = text.strip().lower()
    return s.startswith("may refer to") or s.startswith("can refer to")

def resolve_disambiguation(text, k=3):
    lines = text.split("\n")
    cands = list({c for ln in lines for c in re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", ln)})[:k]
    results = []
    for cand in cands:
        results.extend(fuzzy_lookup(cand, k=1))
    return results

def lookup_entity(ent, k=3):
    direct = search_cached(ent)
    if direct:
        title, text, _ = direct[0]
        if is_disambiguation(text):
            return resolve_disambiguation(text, k)
        return direct
    return fuzzy_lookup(ent, k)

def answer_query(query, k=3):
    for cand in extract_candidates(query):
        res = lookup_entity(cand, k)
        if res:
            return res
    return []

if __name__ == "__main__":
    while True:
        try:
            q = input(">> ").strip()
            if not q:
                break
            hits = answer_query(q, k=3)
            for title, text, score in hits:
                print(f"- {title} (score={score})\n{text[:400]}\n")
        except KeyboardInterrupt:
            break

```

### WIKI.SEARCH.INDEX.7

```python
from datasets import load_dataset
from transformers import pipeline
from rapidfuzz import process, fuzz
import os, pickle, re, sys

# --- Load manageable slice ---
wiki = load_dataset("jordiclive/wikipedia-summary-dataset", split="train")
titles = wiki["title"]
texts = wiki["summary"]

from tqdm import tqdm

if os.path.exists("titles.cache"):
    with open("titles.cache", "rb") as f:
        title_map = pickle.load(f)
else:
    print("Building title map...")
    title_map = {t.lower(): i for i, t in enumerate(tqdm(titles, desc="Indexing titles"))}
    with open("titles.cache", "wb") as f:
        pickle.dump(title_map, f)

# --- Pipelines ---
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
pos_tagger = pipeline(
    "token-classification",
    model="vblagoje/bert-english-uncased-finetuned-pos",
    aggregation_strategy="simple"
)
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Extract candidate search terms ---
def extract_candidates(query):
    ents = [tok["word"] for tok in ner(query)]
    if ents:
        return ents
    tokens = pos_tagger(query)
    phrases, current = [], []
    for tok in tokens:
        if tok["entity_group"] in ["NOUN", "PROPN", "ADJ"]:
            current.append(tok["word"])
        elif tok["word"].lower() in ["of", "in", "on", "for"] and current:
            current.append(tok["word"])
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))
    return phrases or [query]

# --- Fuzzy title match ---
def fuzzy_lookup(ent, k=3, min_score=85):
    matches = process.extract(ent.lower(), list(title_map.keys()), scorer=fuzz.WRatio, limit=k)
    results = []
    for match_title, score, _ in matches:
        if score >= min_score:
            idx = title_map[match_title]
            results.append((titles[idx], texts[idx], score))
    return results

qa = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_from_context(question, context):
    prompt = f"Answer the question based on the text:\n\n{context}\n\nQuestion: {question}"
    return qa(prompt, max_new_tokens=128)[0]["generated_text"]

# --- Main logic ---
def answer_query(question, k=3):
    # pick first relevant entity or noun phrase
    candidates = extract_candidates(question)
    for cand in candidates:
        hits = fuzzy_lookup(cand, k=k)
        if not hits:
            continue
        # choose top hit
        title, text, score = hits[0]
        context = text[:512]
        answer = ask_from_context(question, context)
        return answer, title
    return "No result found.", None

# --- Entry point ---
if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else input(">> ").strip()
    answer, source = answer_query(q)
    print(f"Answer: {answer}")
    if source:
        print(f"(From: {source})")
```
