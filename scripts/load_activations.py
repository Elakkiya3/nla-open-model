import numpy as np

path = "data/sample_activations.npy"

activations = np.load(path)

print("Activation tensor shape:")
print(activations.shape)

print("\nFirst token first 10 neurons:")
print(activations[0, 0, :10])