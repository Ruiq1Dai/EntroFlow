# from modelscope import snapshot_download

# model_dir = snapshot_download(
#     model_id="cross-encoder/nli-deberta-v3-large",
#     local_dir="/data/models/deberta-v3-large-mnli"
# )

# print("模型下载完成：", model_dir)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = "/data/models/deberta-v3-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.eval()

pairs = [
    (
        "A man is eating pizza.",
        "A man is eating food.",
        "expected: entailment"
    ),
    (
        "A man is eating pizza.",
        "Nobody is eating anything.",
        "expected: contradiction"
    ),
    (
        "A man is eating pizza.",
        "The man is wearing a blue shirt.",
        "expected: neutral"
    ),
]

for premise, hypothesis, expected in pairs:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    print("\nPremise:", premise)
    print("Hypothesis:", hypothesis)
    print(expected)

    for i, p in enumerate(probs):
        print(f"{model.config.id2label[i]:14s}: {p.item():.4f}")
