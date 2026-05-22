import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Machine learning helps researchers understand neural networks."

inputs = tokenizer(text, return_tensors="pt")

tokens_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

activations = np.load("data/sample_activations.npy")

tokens = activations[0]

print("Tokens:")
print(tokens_text)

similarity_matrix = cosine_similarity(tokens)

print("\nSimilarity matrix:\n")

for i, token_i in enumerate(tokens_text):
    for j, token_j in enumerate(tokens_text):
        score = similarity_matrix[i][j]
        print(f"{token_i:15} <-> {token_j:15} : {score:.4f}")