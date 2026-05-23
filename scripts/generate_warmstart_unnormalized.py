"""
Warm-Start Data Generation - NO NORMALIZATION
Uses original activation magnitudes
"""

import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = Path("data/warmstart_unnormalized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

texts = [
    # Programming (20)
    "Python is widely used for machine learning and data science.",
    "JavaScript enables interactive web browser applications.",
    "Java uses object-oriented principles for enterprise software.",
    "C++ provides low-level memory control and performance.",
    "Go is optimized for concurrent programming.",
    "Rust ensures memory safety without garbage collection.",
    "Ruby prioritizes developer happiness with elegant syntax.",
    "PHP powers dynamic web server and content management.",
    "Swift is Apple's modern language for iOS development.",
    "Kotlin interoperates seamlessly with Java on JVM.",
    "Algorithms optimize computational efficiency systematically.",
    "Data structures organize information for efficient access.",
    "Functions modularize code into reusable components.",
    "Classes organize data and methods logically.",
    "Debugging identifies and fixes logical errors.",
    "Testing verifies software correctness.",
    "Version control manages code changes.",
    "APIs enable communication between software systems.",
    "Databases store structured information efficiently.",
    "Networking connects computers for communication.",
    
    # Science (30)
    "Quantum computing exploits superposition for parallel computation.",
    "CRISPR gene editing enables precise genetic modification.",
    "Climate modeling predicts future weather patterns.",
    "Nanotechnology manipulates matter at atomic scale.",
    "Synthetic biology engineers biological systems.",
    "Optogenetics uses light to control neural activity.",
    "Stem cells can differentiate into specialized types.",
    "Immunotherapy harnesses immune systems against cancer.",
    "Microbiome research reveals bacterial communities.",
    "Astrobiology searches for life beyond Earth.",
    "Photosynthesis converts light energy into chemical energy.",
    "DNA encodes genetic instructions for organisms.",
    "Proteins perform most biological functions.",
    "Evolution by natural selection explains life diversity.",
    "The immune system defends against pathogens.",
    "Mitochondria produce cellular energy as ATP.",
    "Genes are DNA segments encoding proteins.",
    "Cellular respiration extracts energy from glucose.",
    "Neurotransmitters relay signals between neurons.",
    "Ecosystems involve complex organism interactions.",
    "Atoms are composed of protons, neutrons, electrons.",
    "Chemical bonds connect atoms through electrons.",
    "Molecules form when atoms bond.",
    "Reactions rearrange bonds and transform structure.",
    "Acids and bases show opposite properties.",
    "Oxidation involves loss of electrons.",
    "Catalysts speed up reactions without consumption.",
    "Polymers are long chains of repeating units.",
    "Solutions form when substances dissolve.",
    "Valence electrons determine bonding patterns.",
] + [
    # Physics (20)
    "Quantum mechanics describes particle behavior.",
    "Relativity links space, time, and gravity.",
    "Thermodynamics governs energy transfer.",
    "Newton's laws describe forces and motion.",
    "Electromagnetism unifies electricity and magnetism.",
    "String theory proposes particles as vibrating strings.",
    "Black holes represent extreme gravity.",
    "Particle physics studies fundamental matter.",
    "Astrophysics investigates stars and galaxies.",
    "Cosmology explores universe origin and structure.",
    "Waves propagate through media.",
    "Light exhibits both wave and particle properties.",
    "Sound travels through vibration in media.",
    "Energy exists in multiple forms.",
    "Momentum describes motion quantity.",
    "Gravity attracts massive objects.",
    "Magnetism involves aligned electron spins.",
    "Entropy measures disorder in systems.",
    "Temperature indicates molecular motion.",
    "Pressure is force distributed over area.",
] + [
    # History (20)
    "The Industrial Revolution mechanized production.",
    "The Renaissance revival of classical learning.",
    "World War II defined modern geopolitics.",
    "The French Revolution established democracy.",
    "Ancient Rome developed legal systems.",
    "The printing press enabled mass distribution.",
    "The American Revolution established independence.",
    "The fall of Berlin Wall ended Cold War.",
    "Medieval feudalism structured society.",
    "The Silk Road connected continents.",
    "Ancient Egypt built monuments and systems.",
    "Greek philosophy influenced Western thought.",
    "The Enlightenment promoted reason and science.",
    "The Victorian era shaped modern society.",
    "The Renaissance produced great artists.",
    "Colonialism expanded European influence.",
    "The Industrial Revolution created factories.",
    "The Roaring Twenties changed culture.",
    "The Great Depression caused economic hardship.",
    "The Civil Rights Movement fought discrimination.",
]

print("=" * 70)
print("GENERATING WARM-START DATA - NO NORMALIZATION (210 PAIRS)")
print("=" * 70)

print(f"\n1. Prepared {len(texts)} texts")
print(f"\n2. Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    output_hidden_states=True
)
model.eval()

print(f"   ✓ Model loaded")

print(f"\n3. Extracting activations (NO NORMALIZATION)...")

activations = []

with torch.no_grad():
    for i, text in enumerate(tqdm(texts, desc="Processing")):
        try:
            tokens = tokenizer(
                text,
                max_length=256,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"]
            outputs = model(input_ids, output_hidden_states=True)
            
            hidden = outputs.hidden_states[-1]
            final_token = hidden[:, -1, :]
            
            # NO NORMALIZATION - keep original magnitude
            activation = final_token.cpu().numpy().squeeze()
            activations.append(activation)
        
        except Exception as e:
            print(f"   Error {i}: {e}")

activations = np.array(activations)

print(f"\n4. Saving...")

np.save(OUTPUT_DIR / "activations.npy", activations)

summaries = {str(i): texts[i][:80] + "..." for i in range(len(texts))}
with open(OUTPUT_DIR / "summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)

print(f"\n✅ COMPLETE!")
print(f"   Generated: {len(activations)} pairs")
print(f"   Shape: {activations.shape}")
print(f"   Activation stats:")
print(f"     Mean: {np.mean(activations):.6f}")
print(f"     Std:  {np.std(activations):.6f}")
print(f"     Min:  {np.min(activations):.6f}")
print(f"     Max:  {np.max(activations):.6f}")
print(f"   Saved to: {OUTPUT_DIR}/")
