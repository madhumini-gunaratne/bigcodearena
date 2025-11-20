# Step-by-Step Implementation Guide for BigCodeArena

## Quick Overview of KodCode's Approach

```
INPUT: Existing Questions (from AutoCodeArena)
   ↓
STEP 1: Use LLM + Prompt Templates to Generate New Questions
   ↓
STEP 2: Remove Duplicates (using semantic similarity)
   ↓
OUTPUT: New, Diverse Questions
```

**Key Difference from your original plan:**
- ❌ NOT about "enhancing responses" or improving code quality
- ✅ ABOUT "generating new/diverse questions" from existing ones

---

## Where to Create Files

### Directory Structure to Create

```
autocodearena/
├── question_generator/              ← NEW FOLDER (already created)
│   ├── __init__.py
│   ├── KODCODE_ANALYSIS.md         ← Reference document
│   ├── README.md                    ← Setup instructions
│   │
│   ├── 1_generate_from_seed.py      ← STEP 1: Generate new questions
│   ├── 2_diversify_questions.py     ← STEP 2: Create variations
│   ├── 3_sanitize_questions.py      ← STEP 3: Remove duplicates
│   │
│   ├── prompts/                     ← Prompt templates
│   │   ├── gen_from_seed.md
│   │   ├── diversify_v1.md
│   │   └── diversify_v2.md
│   │
│   ├── seeds/                       ← Seed data (existing questions)
│   │   ├── web_design_questions.jsonl
│   │   ├── game_dev_questions.jsonl
│   │   └── problem_solving_questions.jsonl
│   │
│   ├── config/
│   │   └── generation_config.yaml
│   │
│   ├── results/                     ← Output generated questions
│   │   ├── generated_questions.jsonl
│   │   ├── diversified_questions.jsonl
│   │   └── final_questions.jsonl
│   │
│   └── utils/
│       ├── llm_client.py           ← Call LLM (vLLM or API)
│       └── semantic_dedup.py       ← Deduplication logic
```

---

## How the Process Works - Step by Step

### **STEP 1: Prepare Seed Data**

**What:** Load existing AutoCodeArena questions as "seeds"

**Where:** `autocodearena/question_generator/seeds/`

**How:**
```python
# Load from HuggingFace dataset
from datasets import load_dataset

dataset = load_dataset("bigcode/autocodearena-v0")
questions = dataset['train']

# Filter by category
web_design_qs = [q for q in questions if q['category'] == 'web_design']

# Save as seed file (JSONL format)
import jsonlines
with jsonlines.open('seeds/web_design_questions.jsonl', 'w') as writer:
    for q in web_design_qs:
        writer.write({
            'uid': q['uid'],
            'instruction': q['instruction'],
            'category': q['category']
        })
```

**Output:** `seeds/web_design_questions.jsonl`
```json
{"uid": "uid_123", "instruction": "Design an HTML5 canvas...", "category": "web_design"}
{"uid": "uid_456", "instruction": "Create a responsive grid...", "category": "web_design"}
{"uid": "uid_789", "instruction": "Build a color picker...", "category": "web_design"}
```

---

### **STEP 2: Create Prompt Templates**

**Where:** `autocodearena/question_generator/prompts/`

#### **File 1: `gen_from_seed.md`** - Generate from similar questions
```markdown
# Generate New Coding Assessment Question

## Task
Analyze the provided coding questions and generate a new question that follows the same pattern.

## Objective
Create a new question that:
- Matches the style and tone of existing questions
- Has similar difficulty level
- Covers related but different topics
- Is unique and not a simple rephrasing

## Sample Questions
{SAMPLE_QUESTIONS}

## Guidelines
- Maintain consistency in language and format
- Match the technical depth
- Be creative but realistic
- Output ONLY the new question, no explanation

## Output
[New Question Here]
```

#### **File 2: `diversify_v1.md`** - Create variations with different contexts
```markdown
# Create a Diversified Variation of a Coding Question

## Task
Take an existing coding question and create a variation with a different context/scenario
while maintaining the same core programming skills.

## Original Question
{ORIGINAL_QUESTION}

## Objective
Generate ONE variation that:
- Tests the same programming concepts
- Applies to a DIFFERENT real-world context
- INCREASES complexity with new constraints
- Remains solvable with the same technical skills

## Context Ideas
- Change industry (finance → healthcare → e-commerce)
- Change application type (web app → mobile → embedded system)
- Add performance/scale requirements
- Add security/reliability requirements

## Output Format
<|Analysis Begin|>
[Explain how you're maintaining core skills but changing context]
<|Analysis End|>

<|Question Begin|>
[The new diversified question]
<|Question End|>
```

#### **File 3: `diversify_v2.md`** - Increase complexity
```markdown
# Increase Question Complexity

## Original Question
{ORIGINAL_QUESTION}

## Task
Create a harder version of this question by:
- Adding more constraints
- Requiring optimization
- Including edge cases
- Combining multiple concepts
- Requiring system design thinking

## Output
[More complex version of the question]
```

---

### **STEP 3: Write Generation Script**

**File:** `autocodearena/question_generator/1_generate_from_seed.py`

```python
import json
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from jinja2 import Template

def generate_from_seed(seed_file, prompt_template, num_questions, model_name="qwen2.5-7b-inst"):
    """
    Generate new questions from seed questions.
    
    Input: seed file with existing questions
    Process: Load samples → Create prompt → Call LLM → Extract output
    Output: JSONL file with generated questions
    """
    
    # 1. Load seed questions
    seeds = []
    with open(seed_file) as f:
        for line in f:
            seeds.append(json.loads(line))
    
    # 2. Initialize LLM
    llm = LLM(model=model_name, max_model_len=4096, dtype="float16")
    sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=2048)
    
    # 3. Load prompt template
    with open(prompt_template) as f:
        template_str = f.read()
    template = Template(template_str)
    
    # 4. Generate new questions
    generated_questions = []
    
    for i in range(num_questions):
        # Pick 2-3 random seed questions
        sample_qs = seeds[i % len(seeds)]  # Simple rotation; use random in production
        
        # Format seed questions for prompt
        sample_text = "\n\n".join([f"[Question {j+1}]: {q['instruction']}" 
                                   for j, q in enumerate(seeds[i:i+2])])
        
        # Create prompt
        prompt = template.render(SAMPLE_QUESTIONS=sample_text)
        
        # Call LLM
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Extract question (remove metadata like "Question X:" if present)
        question = extract_instruction(generated_text)
        
        generated_questions.append({
            "uid": f"gen_{i:06d}",
            "instruction": question,
            "category": seeds[i % len(seeds)]['category'],
            "source_seeds": [seeds[i % len(seeds)]['uid']],
            "generation_method": "from_seed"
        })
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{num_questions} questions")
    
    # 5. Save output
    output_file = Path("results/generated_questions.jsonl")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        for q in generated_questions:
            f.write(json.dumps(q) + '\n')
    
    print(f"Saved {len(generated_questions)} questions to {output_file}")

def extract_instruction(text):
    """Remove question numbering and metadata"""
    import re
    # Remove patterns like "Question X:", "[Question X]:", etc.
    text = re.sub(r'^\[?Question\s*\d*\]?:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^#+\s*', '', text)  # Remove markdown headers
    return text.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_file", default="seeds/web_design_questions.jsonl")
    parser.add_argument("--prompt_template", default="prompts/gen_from_seed.md")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--model", default="qwen2.5-7b-inst")
    args = parser.parse_args()
    
    generate_from_seed(args.seed_file, args.prompt_template, args.num_questions, args.model)
```

**Usage:**
```bash
cd autocodearena/question_generator
python 1_generate_from_seed.py \
    --seed_file seeds/web_design_questions.jsonl \
    --num_questions 100 \
    --model qwen2.5-7b-inst
```

---

### **STEP 4: Write Deduplication Script**

**File:** `autocodearena/question_generator/3_sanitize_questions.py`

```python
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_questions(input_file, output_file, similarity_threshold=0.85):
    """
    Remove near-duplicate questions using semantic similarity.
    
    Process:
    1. Load all questions
    2. Encode to embeddings
    3. Find similar pairs
    4. Keep only unique ones
    """
    
    # 1. Load questions
    questions = []
    with open(input_file) as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions")
    
    # 2. Encode to embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    instructions = [q['instruction'] for q in questions]
    embeddings = model.encode(instructions, batch_size=32, show_progress_bar=True)
    
    # 3. Find duplicates
    similarity_matrix = cosine_similarity(embeddings)
    
    # Keep track of which questions to keep
    keep_indices = set(range(len(questions)))
    
    for i in range(len(questions)):
        if i not in keep_indices:
            continue
        
        for j in range(i+1, len(questions)):
            if j not in keep_indices:
                continue
            
            if similarity_matrix[i][j] > similarity_threshold:
                # Remove the later one
                keep_indices.discard(j)
                print(f"Removing duplicate: Q{j} (similar to Q{i}, similarity={similarity_matrix[i][j]:.3f})")
    
    # 4. Save unique questions
    unique_questions = [questions[i] for i in sorted(keep_indices)]
    
    with open(output_file, 'w') as f:
        for q in unique_questions:
            f.write(json.dumps(q) + '\n')
    
    print(f"Kept {len(unique_questions)} unique questions out of {len(questions)}")
    print(f"Removed {len(questions) - len(unique_questions)} duplicates")

if __name__ == "__main__":
    deduplicate_questions(
        input_file="results/generated_questions.jsonl",
        output_file="results/final_questions.jsonl",
        similarity_threshold=0.85
    )
```

---

## Complete Workflow

```bash
# 1. Prepare seeds from existing AutoCodeArena questions
# (you would create a script that loads from HuggingFace and saves to seeds/)

# 2. Generate new questions from seeds
python 1_generate_from_seed.py \
    --seed_file seeds/web_design_questions.jsonl \
    --num_questions 100

# 3. (Optional) Create variations/diversifications
python 2_diversify_questions.py \
    --input_file results/generated_questions.jsonl \
    --num_variations 2

# 4. Deduplicate
python 3_sanitize_questions.py \
    --input_file results/diversified_questions.jsonl \
    --output_file results/final_questions.jsonl

# 5. Result: final_questions.jsonl with new, high-quality, diverse questions
```

---

## Key Files to Create Summary

| File | Purpose | Status |
|------|---------|--------|
| `1_generate_from_seed.py` | Generate questions | Need to create |
| `2_diversify_questions.py` | Create variations | Need to create |
| `3_sanitize_questions.py` | Remove duplicates | Need to create |
| `prompts/gen_from_seed.md` | Generation prompt | Need to create |
| `prompts/diversify_v1.md` | Diversification prompt | Need to create |
| `seeds/` | Store seed questions | Need to create |
| `results/` | Store generated questions | Need to create |

---

## Data Flow Example

```
SEEDS (loaded from HuggingFace)
├─ uid_001: "Design an HTML canvas drawing app"
├─ uid_002: "Create a responsive grid layout"
└─ uid_003: "Build a color picker widget"

GENERATION (step 1)
├─ gen_000001: "Design a collaborative whiteboard application"
├─ gen_000002: "Build an SVG-based diagram editor"
└─ gen_000003: "Create a pixel art editor with layers"

DIVERSIFICATION (step 2)
├─ gen_000001_v1: "Design a real-time multiplayer drawing platform"
└─ gen_000001_v2: "Build a medical imaging annotation tool"

DEDUPLICATION (step 3)
├─ Final set: 28 unique questions (removed 4 near-duplicates)
```

---

## What NOT to Do

❌ Don't try to improve existing code/answers  
❌ Don't generate test cases  
❌ Don't focus on response quality enhancement  
✅ DO focus on: **Generating new, diverse questions from existing ones**

---

## Next Steps

1. Create the folder structure
2. Create prompt templates
3. Implement `1_generate_from_seed.py`
4. Extract seeds from AutoCodeArena dataset
5. Run generation pipeline
6. Evaluate generated questions

Would you like me to create these files now?
