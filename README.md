# wiki.search

```python
from datasets import load_dataset
import pickle, os, sys
from transformers import pipeline
from tqdm import tqdm

wiki = load_dataset("jordiclive/wikipedia-summary-dataset", split="train")
titles = wiki["title"]
texts = wiki["summary"]

cache_path = "titles.cache"
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        title_index = pickle.load(f)
else:
    print("Building title map...")
    title_index = {t.lower(): i for i, t in enumerate(tqdm(titles, desc="Indexing titles"))}
    with open(cache_path, "wb") as f:
        pickle.dump(title_index, f)

def find_exact_title(title: str):
    t = title.lower()
    return title_index.get(t)

# question → extract subject
question = sys.argv[1]
ner = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")
entities = ner(question)
subjects = [e["word"] for e in entities if e["entity_group"] in ["PER","ORG","LOC","MISC"]] or [question.split()[0]]
subject = subjects[0].strip()

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# use the exact title if available
idx = find_exact_title(subject)
if idx is not None:
    contexts = [(subject, idx)]
else:
    # fallback to partial matches
    contexts = [(t, i) for t, i in title_index.items() if subject.lower() in t][:5]

results = []
for title, idx in contexts:
    context = wiki[idx]["summary"]
    r = qa(question=question, context=context)
    results.append({
        "title": title,
        "subject": subject,
        "context": context,
        "answer": r["answer"],
        "score": r["score"]
    })

best = sorted(results, key=lambda x: x["score"], reverse=True)[0]
print(f"{best['title']} ({best['subject']}): {best['answer']} (score={best['score']:.3f})\n{best['context']}")


# after selecting 'best'
context = best["context"]
question = sys.argv[1]

gen = pipeline("text2text-generation", model="google/flan-t5-large")

prompt = f"Answer the question based on this text:\n\n{context}\n\nQuestion: {question}"
answer = gen(prompt, max_new_tokens=64)[0]["generated_text"]

print(f"{best['title']}: {answer}")```
