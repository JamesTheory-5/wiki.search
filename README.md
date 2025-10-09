# wiki.search

```python
from datasets import load_dataset
import pickle, os, sys
from transformers import pipeline
from tqdm import tqdm

# Load dataset
wiki = load_dataset("wikipedia", "20220301.en", split="train")
titles = wiki["title"]
texts = wiki["text"]


# Cache title index
cache_path = "titles.cache"
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        title_index = pickle.load(f)
else:
    print("Building title map...")
    title_index = {t.lower(): i for i, t in enumerate(tqdm(titles, desc="Indexing titles"))}
    with open(cache_path, "wb") as f:
        pickle.dump(title_index, f)


# Extract subject from question
question = sys.argv[1]
ner = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")
entities = ner(question)
subjects = [e["word"] for e in entities if e["entity_group"] in ["PER","ORG","LOC","MISC"]] or [question.split()[0]]
subject = subjects[0].strip()


def subject_indexes(subject: str):
    """Return up to n (title, index) pairs for a given subject name."""
    s = subject.lower()
    ban_prefix = ("category:", "list of", "template:", "file:", "portal:")
    matches = [
        (t, i) for t, i in title_index.items()
        if s == t and not t.startswith(ban_prefix)
    ]
    # prioritize exact match
    matches.sort(key=lambda x: (x[0] != s))
    return matches

# Get contexts
contexts = subject_indexes(subject)
print(f"Found {len(contexts)} contexts for '{subject}'")

results = []
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
for title, idx in contexts:
    context = wiki[idx]["text"]
    r = qa(question=question, context=context)
    results.append({
        "title": title,
        "subject": subject,
        "context": context.split("\n")[0],
        "answer": r["answer"],
        "score": r["score"]
    })

if not results:
    print("No valid results found.")
    sys.exit()

# Pick best
best = sorted(results, key=lambda x: x["score"], reverse=True)[0]
print(f"{best['title']} ({best['subject']}): {best['answer']} (score={best['score']:.3f})")
print(f"Context: {best['context']}\n")

# Generative refinement
gen = pipeline("text2text-generation", model="google/flan-t5-large")
prompt = f"Answer the question based on this text:\n\n{best['context']}\n\nQuestion: {question}"
answer = gen(prompt, max_new_tokens=64)[0]["generated_text"]

print(f"{best['title']}: {answer}")

```
