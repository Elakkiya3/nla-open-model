"""
Generate warm-start data WITHOUT using Claude API.
Uses simple rule-based summaries instead.
Still produces valid training data.
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = -1
OUTPUT_DIR = Path("data/warmstart")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

def generate_simple_summary(text):
    """Generate a simple summary without API."""
    words = text.split()
    key_words = [w for w in words if len(w) > 4][:5]
    concepts = {
        "machine": "computational systems",
        "learning": "pattern recognition",
        "data": "information processing",
        "neural": "network architecture",
        "model": "predictive framework",
        "algorithm": "computational procedure",
        "physics": "physical laws",
        "chemistry": "molecular reactions",
        "biology": "living systems",
        "history": "temporal narrative",
        "art": "creative expression",
        "music": "harmonic patterns",
    }
    
    concept_list = []
    for word in key_words:
        for key, value in concepts.items():
            if key in word.lower():
                concept_list.append(value)
                break
    
    if not concept_list:
        concept_list = ["general knowledge", "topic understanding"]
    
    summary = f"The text discusses: {', '.join(concept_list[:3])}. "
    summary += "Key themes include domain-specific concepts and foundational knowledge. "
    summary += "The reader should anticipate related terminology and connected ideas."
    
    return summary

def extract_activation(text, model, tokenizer, layer_idx=-1):
    """Extract activation at final token."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    activation = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
    return activation

def create_sample_texts():
    """Create diverse sample texts."""
    texts = [
        "Machine learning has revolutionized how we process data. Neural networks can learn patterns from large datasets.",
        "Artificial intelligence is transforming industries. Computer vision helps robots understand their environment.",
        "Data science combines statistics and programming. Big data analytics reveals hidden patterns.",
        "Physics describes the fundamental laws of nature. Quantum mechanics governs subatomic particles.",
        "Chemistry studies the properties of matter and reactions. Atoms bond together to form molecules.",
        "Biology is the study of living organisms. Evolution explains the diversity of life.",
        "The Renaissance marked a revival of classical learning. Medieval Europe was a period of feudalism.",
        "Ancient Rome built an empire spanning continents. Greek philosophy influenced Western thought.",
        "Shakespeare wrote sonnets and plays that endure centuries. Poetry uses metaphor and imagery.",
        "Fantasy worlds capture imagination with magic and adventure. Science fiction explores future possibilities.",
        "Supply and demand determine market prices. Economics studies how societies allocate resources.",
        "Trade between nations benefits both economies. Interest rates influence borrowing and investment.",
        "Mountains form through tectonic plate movements. Rivers shape landscapes as they flow to the sea.",
        "Deserts receive little rainfall yet support adapted life. Oceans regulate Earth's temperature.",
        "Memory stores information through neural connections. Emotions influence decision making.",
        "Learning occurs through repeated practice and feedback. Motivation drives human behavior.",
        "The immune system protects against diseases. Antibiotics kill bacteria that cause infections.",
        "The brain controls movement and thought. Neurons transmit signals throughout the nervous system.",
        "Painting uses color and form to create visual art. Sculpture shapes three-dimensional materials.",
        "Music combines rhythm, melody, and harmony. Dance expresses emotion through movement.",
        # Mathematics
        "Linear algebra studies vectors, matrices, and transformations. Calculus analyzes rates of change and accumulation.",
        "Probability theory models uncertainty and randomness. Statistics helps interpret data distributions.",
        "Number theory investigates properties of integers. Prime numbers play a central role in cryptography.",
        "Topology studies geometric properties preserved under deformation. Differential equations describe dynamic systems.",

        # Programming
        "Python is widely used for machine learning and automation. Functions and classes improve code organization.",
        "Algorithms optimize computational efficiency. Data structures enable fast retrieval and storage.",
        "Software engineering focuses on scalable system design. Debugging helps identify logical errors in programs.",
        "Databases store structured information efficiently. SQL enables querying relational datasets.",

        # Astronomy
        "Galaxies contain billions of stars bound by gravity. Black holes distort spacetime through intense mass.",
        "The solar system includes planets, moons, and asteroids. Telescopes help scientists observe distant objects.",
        "Supernova explosions distribute heavy elements across space. Cosmology studies the origin of the universe.",
        "Dark matter influences galactic motion despite being invisible. Exoplanets orbit stars beyond our solar system.",

        # Linguistics
        "Linguistics studies language structure and communication. Syntax governs sentence organization.",
        "Phonetics analyzes speech sounds and pronunciation. Semantics examines meaning in language.",
        "Languages evolve over time through cultural interaction. Translation requires contextual understanding.",
        "Grammar rules shape how sentences are interpreted. Pragmatics studies language use in social situations.",

        # Neuroscience
        "Neuroscience investigates how the brain processes information. Synapses transmit signals between neurons.",
        "Memory formation depends on neural plasticity. Cognitive science explores perception and reasoning.",
        "The cerebral cortex supports higher-order thinking. Dopamine influences motivation and reward systems.",
        "Brain imaging technologies reveal patterns of neural activity. Sleep is important for memory consolidation.",

        # Philosophy
        "Philosophy examines knowledge, ethics, and existence. Logic provides rules for valid reasoning.",
        "Ancient philosophers debated the nature of reality. Moral philosophy studies principles of right and wrong.",
        "Existentialism explores freedom and human meaning. Epistemology investigates how knowledge is acquired.",
        "Metaphysics studies causality and the structure of reality. Critical thinking evaluates assumptions systematically.",

        # Law
        "Constitutional law defines the structure of government. Courts interpret legal principles and precedents.",
        "Criminal law addresses offenses against society. Evidence is essential during legal proceedings.",
        "International law governs relations between nations. Human rights law protects individual freedoms.",
        "Contracts establish enforceable agreements between parties. Intellectual property protects creative works.",

        # Robotics
        "Robotics combines mechanics, electronics, and artificial intelligence. Sensors help robots navigate environments.",
        "Autonomous systems make decisions without human intervention. Industrial robots improve manufacturing efficiency.",
        "Computer vision enables robots to recognize objects. Reinforcement learning improves robotic control policies.",
        "Humanoid robots imitate human movement patterns. Motion planning algorithms optimize navigation paths.",

        # Cybersecurity
        "Cybersecurity protects systems from digital attacks. Encryption secures sensitive information.",
        "Network security prevents unauthorized access to data. Firewalls monitor incoming and outgoing traffic.",
        "Malware can disrupt computer operations and steal information. Authentication systems verify user identity.",
        "Ethical hackers identify vulnerabilities before attackers exploit them. Cyber defense requires continuous monitoring.",

        # Climate Science
        "Climate change affects global temperatures and weather patterns. Greenhouse gases trap heat in the atmosphere.",
        "Renewable energy reduces dependence on fossil fuels. Carbon emissions contribute to environmental warming.",
        "Ocean currents influence regional climates worldwide. Deforestation impacts biodiversity and ecosystems.",
        "Climate models simulate future environmental conditions. Sustainable development balances growth and conservation.",
    ]
    
    expanded = []
    for text in texts:
        sentences = text.split(". ")
        for i in range(1, len(sentences)):
            partial = ". ".join(sentences[:i])
            if len(partial.split()) >= 5:
                expanded.append(partial)
    
    return expanded

def main():
    print("=" * 60)
    print("Generating Warm-Start Data (OFFLINE - No API)")
    print("=" * 60)
    
    print("\n1. Creating sample texts...")
    texts = create_sample_texts()
    random.shuffle(texts)
    expanded = []

    for _ in range(10):
        for text in texts:
            expanded.append(text)

    random.shuffle(expanded)

    texts = expanded[:120]
    print(f"   Created {len(texts)} text samples")
    
    print(f"\n2. Loading model...")
    
    warmstart_pairs = []
    
    print(f"\n3. Extracting activations and generating summaries...")
    print(f"   Processing {len(texts)} texts...\n")
    
    for i, text in enumerate(texts):
        try:
            activation = extract_activation(text, model, tokenizer, LAYER_IDX)
            summary = generate_simple_summary(text)
            
            warmstart_pairs.append({
                "activation": activation.tolist(),
                "summary": summary,
                "text": text[:200]
            })
            
            if (i + 1) % 50 == 0:
                print(f"   ✓ Processed {i+1}/{len(texts)} pairs")
        
        except Exception as e:
            print(f"   ✗ Error on {i}: {e}")
            continue
    
    print(f"\n4. Saving data...")
    
    output_file = OUTPUT_DIR / "warmstart_pairs.json"
    with open(output_file, "w") as f:
        json.dump(warmstart_pairs, f, indent=2)
    
    activations = np.array([p["activation"] for p in warmstart_pairs])
    summaries = [p["summary"] for p in warmstart_pairs]
    
    np.save(OUTPUT_DIR / "activations.npy", activations)
    with open(OUTPUT_DIR / "summaries.json", "w") as f:
        json.dump(summaries, f)
    
    print(f"\n✅ Warm-Start Data Generation Complete!")
    print(f"   Pairs generated: {len(warmstart_pairs)}")
    print(f"   Activation shape: {activations.shape}")
    print(f"   Files saved to: {OUTPUT_DIR}/")
    print(f"\n   Ready for training:")
    print(f"   - python scripts/train_av_warmstart.py")
    print(f"   - python scripts/train_ar_warmstart.py")

if __name__ == "__main__":
    main()
