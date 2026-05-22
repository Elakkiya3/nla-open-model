import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Machine learning helps researchers understand neural networks."

inputs = tokenizer(text, return_tensors="pt")

tokens_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

activations = np.load("data/sample_activations.npy")

tokens = activations[0]

# Reduce 896D -> 2D
pca = PCA(n_components=2)

reduced = pca.fit_transform(tokens)

# Plot
plt.figure(figsize=(8, 6))

for i, token in enumerate(tokens_text):
    x = reduced[i, 0]
    y = reduced[i, 1]

    plt.scatter(x, y)
    plt.text(x + 0.02, y + 0.02, token)

plt.title("Token Representation Geometry (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.grid(True)

plt.show()