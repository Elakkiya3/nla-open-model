"""
Large-scale warm-start data: 290 diverse activation-summary pairs
FIXED: No manual device movement with device_map="auto"
"""

import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = Path("data/warmstart_large")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EXPANDED TEXT CORPUS (290 diverse samples)
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
    
    # History (25)
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
    "Ancient China developed advanced technology.",
    "The Ottoman Empire spanned continents.",
    "The Reformation challenged religious authority.",
    "The Scientific Revolution established empiricism.",
    "World War One reshaped geopolitics.",
    
    # Mathematics (25)
    "Calculus analyzes continuous change.",
    "Linear algebra studies vectors and matrices.",
    "Probability quantifies uncertainty.",
    "Graph theory studies networks.",
    "Abstract algebra formalizes operations.",
    "Topology examines deformation properties.",
    "Number theory investigates integers.",
    "Combinatorics counts arrangements.",
    "Differential equations model change.",
    "Statistics provides data inference.",
    "Geometry studies shapes and spaces.",
    "Logic establishes reasoning rules.",
    "Set theory formalizes collections.",
    "Category theory unifies structures.",
    "Functional analysis studies function spaces.",
    "Complex analysis extends calculus.",
    "Real analysis formalizes calculus.",
    "Algebra studies symbolic manipulation.",
    "Trigonometry relates angles and sides.",
    "Fourier analysis decomposes signals.",
    "Numerical analysis approximates solutions.",
    "Discrete mathematics studies finite systems.",
    "Game theory analyzes strategic interaction.",
    "Optimization finds best solutions.",
    "Approximation theory bounds errors.",
    
    # Philosophy (20)
    "Epistemology examines knowledge nature.",
    "Metaphysics investigates reality.",
    "Ethics determines right behavior.",
    "Existentialism emphasizes freedom.",
    "Logic establishes reasoning rules.",
    "Aesthetics analyzes beauty.",
    "Political philosophy examines governance.",
    "Phenomenology studies consciousness.",
    "Pragmatism evaluates practical consequences.",
    "Skepticism questions knowledge certainty.",
    "Stoicism teaches virtue acceptance.",
    "Utilitarianism maximizes happiness.",
    "Kantian ethics emphasizes duty.",
    "Contractarianism bases rules on agreement.",
    "Virtue ethics emphasizes character.",
    "Care ethics prioritizes relationships.",
    "Marxism critiques class systems.",
    "Feminism addresses gender equality.",
    "Nihilism denies inherent meaning.",
    "Relativism denies absolute truth.",
    
    # Arts (20)
    "Painting combines color and form.",
    "Sculpture shapes three-dimensional materials.",
    "Photography captures light-based images.",
    "Digital art uses computation.",
    "Performance art challenges boundaries.",
    "Drawing creates visual images.",
    "Installation art transforms spaces.",
    "Conceptual art emphasizes ideas.",
    "Abstract art removes recognizable subjects.",
    "Realism depicts accurate subjects.",
    "Impressionism prioritizes light and color.",
    "Expressionism conveys emotion.",
    "Cubism fragments visual perspective.",
    "Surrealism explores unconscious imagery.",
    "Pop art appropriates consumer culture.",
    "Minimalism uses essential forms.",
    "Abstract expressionism emphasizes gesture.",
    "Postmodernism questions grand narratives.",
    "Street art uses public spaces.",
    "Video art combines technology and aesthetics.",
    
    # Music (20)
    "Rhythm organizes sound patterns.",
    "Melody is a note sequence.",
    "Harmony combines simultaneous tones.",
    "Instruments produce sound.",
    "Composition arranges sounds.",
    "Jazz improvisation is spontaneous.",
    "Classical music follows formal structures.",
    "Rock music revolutionized popular culture.",
    "Electronic music uses synthesizers.",
    "Blues expresses emotional experience.",
    "Folk music reflects cultural traditions.",
    "Opera combines music and theater.",
    "Symphony uses orchestral instruments.",
    "Chamber music uses small ensembles.",
    "Concerto features solo instrument.",
    "Sonata has multiple movements.",
    "Fugue uses counterpoint technique.",
    "Chorus provides ensemble harmony.",
    "Orchestra combines instrument sections.",
    "Conducting unifies ensemble performance.",
    
    # Economics (15)
    "Supply and demand determine prices.",
    "Economics studies resource allocation.",
    "Trade benefits both parties.",
    "Interest rates influence borrowing.",
    "Markets allocate goods efficiently.",
    "Inflation reduces purchasing power.",
    "GDP measures economic output.",
    "Unemployment reflects labor conditions.",
    "Wages compensate labor effort.",
    "Capital investments generate returns.",
    "Competition drives efficiency.",
    "Monopolies concentrate market power.",
    "Consumers choose preferred goods.",
    "Producers maximize profit.",
    "Banking systems facilitate transactions.",
    
    # Medicine (15)
    "Cardiology treats heart diseases.",
    "Neurology treats nervous system.",
    "Oncology treats cancer.",
    "Surgery uses intervention methods.",
    "Pharmacology studies drug effects.",
    "Pathology studies disease.",
    "Radiology uses imaging.",
    "Pediatrics treats children.",
    "Geriatrics treats elderly.",
    "Psychiatry treats mental illness.",
    "Physical therapy restores function.",
    "Immunology studies immune systems.",
    "Microbiology studies microorganisms.",
    "Genetics studies heredity.",
    "Epidemiology studies disease spread.",
]

print("=" * 70)
print("GENERATING LARGE-SCALE WARM-START DATA (290 PAIRS)")
print("=" * 70)

print(f"\n1. Prepared {len(texts)} unique texts")

print(f"\n2. Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # Let accelerate handle device placement
    output_hidden_states=True
)
model.eval()

print(f"   ✓ Model loaded with device_map='auto'")

print(f"\n3. Extracting activations...")

activations = []

with torch.no_grad():
    for i, text in enumerate(tqdm(texts, desc="Processing texts")):
        try:
            # Tokenize
            tokens = tokenizer(
                text,
                max_length=256,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"]
            
            # Forward pass (device handled automatically by model)
            outputs = model(input_ids, output_hidden_states=True)
            
            # Get final token from final layer
            hidden = outputs.hidden_states[-1]  # (1, seq_len, 896)
            final_token = hidden[:, -1, :]      # (1, 896)
            
            # Normalize
            final_token = final_token / (torch.norm(final_token) + 1e-8)
            
            activation = final_token.cpu().numpy().squeeze()
            activations.append(activation)
        
        except Exception as e:
            print(f"\n   Error on text {i}: {e}")
            continue

activations = np.array(activations)

print(f"\n4. Saving results...")

# Save activations
np.save(OUTPUT_DIR / "activations.npy", activations)

# Save summaries
summaries = {str(i): texts[i][:80] + "..." for i in range(len(texts))}
with open(OUTPUT_DIR / "summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)

print(f"\n✅ COMPLETE!")
print(f"   Generated: {len(activations)} activations")
print(f"   Shape: {activations.shape}")
print(f"   Saved to: {OUTPUT_DIR}/")
