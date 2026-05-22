"""
Generate (activation, summary) pairs for AV/AR initialization.
Uses local text data instead of downloading from HF Hub.
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from anthropic import Anthropic
import random

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = -1
OUTPUT_DIR = Path("data/warmstart")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
client = Anthropic()

def generate_claude_summary(text_prefix):
    """Use Claude to describe what the model is thinking."""
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Analyze what a language model is tracking internally at the end of this text.

TEXT FRAGMENT:
{text_prefix}

List 4-5 key features the model would use to predict next tokens:
1. Immediate syntactic constraints
2. Semantic patterns
3. Stylistic/register cues
4. Longer-range dependencies
5. Last token's role

Be specific with examples. Keep it concise (150 words max)."""
        }]
    )
    return message.content[0].text

def extract_activation(text, model, tokenizer, layer_idx=-1):
    """Extract activation at final token of given layer."""
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
    """
    Create diverse sample texts locally (no internet needed).
    These are diverse enough for warm-start training.
    """
    texts = [
        # Technology
        "Machine learning has revolutionized how we process data. Neural networks can learn patterns from large datasets. Deep learning uses multiple layers of abstraction.",
        "Artificial intelligence is transforming industries. Computer vision helps robots understand their environment. Natural language processing enables machines to understand human text.",
        "Data science combines statistics and programming. Big data analytics reveals hidden patterns. Cloud computing provides scalable infrastructure for processing.",
        
        # Science
        "Physics describes the fundamental laws of nature. Quantum mechanics governs subatomic particles. Einstein's theory of relativity changed our understanding of space and time.",
        "Chemistry studies the properties of matter and reactions. Atoms bond together to form molecules. Chemical reactions release or absorb energy.",
        "Biology is the study of living organisms. Evolution explains the diversity of life. Ecosystems maintain balance through complex interactions.",
        
        # History
        "The Renaissance marked a revival of classical learning. Medieval Europe was a period of feudalism. The Industrial Revolution transformed manufacturing and society.",
        "Ancient Rome built an empire spanning continents. Greek philosophy influenced Western thought for millennia. The Byzantine Empire preserved classical knowledge.",
        
        # Literature
        "Shakespeare wrote sonnets and plays that endure centuries. Poetry uses metaphor and imagery to convey emotion. Novels tell complex stories through narrative prose.",
        "Fantasy worlds capture imagination with magic and adventure. Science fiction explores possibilities of future technology. Mystery novels keep readers guessing until the end.",
        
        # Economics
        "Supply and demand determine market prices. Economics studies how societies allocate resources. Inflation affects purchasing power and savings.",
        "Trade between nations benefits both economies. Interest rates influence borrowing and investment. Stocks represent ownership in companies.",
        
        # Geography
        "Mountains form through tectonic plate movements. Rivers shape landscapes as they flow to the sea. Deserts receive little rainfall yet support adapted life.",
        "Climate zones distribute around the equator based on latitude. Oceans cover most of Earth's surface and regulate temperature. Forests provide oxygen and absorb carbon dioxide.",
        
        # Psychology
        "Memory stores information through neural connections. Emotions influence decision making and behavior. Stress affects both mental and physical health.",
        "Learning occurs through repeated practice and feedback. Motivation drives human behavior toward goals. Personality traits remain relatively stable over time.",
        
        # Medicine
        "The immune system protects against diseases. Antibiotics kill bacteria that cause infections. Vaccines prevent diseases by training the immune system.",
        "The brain controls movement and thought. Neurons transmit signals throughout the nervous system. Sleep is essential for memory consolidation.",
        
        # Art
        "Painting uses color and form to create visual art. Sculpture shapes three-dimensional materials. Music combines rhythm, melody, and harmony.",
        "Dance expresses emotion through movement. Architecture designs buildings for function and beauty. Photography captures moments in time.",
    ]
    
    # Expand with variations
    expanded = []
    for text in texts:
        sentences = text.split(". ")
        for i in range(1, len(sentences)):
            partial = ". ".join(sentences[:i])
            expanded.append(partial)
    
    return expanded

def main():
    print("=" * 60)
    print("Generating Warm-Start Data (Local Texts)")
    print("=" * 60)
    
    # Create sample texts locally (no internet needed)
    print("\n1. Creating sample texts...")
    texts = create_sample_texts()
    print(f"   Created {len(texts)} text samples")
    
    # Shuffle
    random.shuffle(texts)
    texts = texts[:250]  # Use 250 examples
    
    print(f"\n2. Loading model ({MODEL_NAME})...")
    
    warmstart_pairs = []
    
    print(f"\n3. Generating summaries for {len(texts)} texts...")
    print("   (This will take 10-15 minutes due to API rate limits)\n")
    
    for i, text in enumerate(texts):
        try:
            # Extract activation
            activation = extract_activation(text, model, tokenizer, LAYER_IDX)
            
            # Generate summary using Claude
            summary = generate_claude_summary(text)
            
            warmstart_pairs.append({
                "activation": activation.tolist(),
                "summary": summary,
                "text": text[:200]
            })
            
            if (i + 1) % 10 == 0:
                print(f"   ✓ Generated {i+1}/{len(texts)} pairs")
                # Show a sample summary
                print(f"     Sample summary: {summary[:100]}...")
        
        except Exception as e:
            print(f"   ✗ Error on example {i}: {e}")
            continue
    
    # Save results
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