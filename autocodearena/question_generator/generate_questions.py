"""
Step 2: Generate New Questions from Seeds

This script reads seed questions and uses an LLM to generate new questions
based on the pattern/style of the seeds.

It uses configuration from generation_config.yaml and the prompt template
from prompts/gen_from_seed.md

"""

import json
import argparse
import yaml
from pathlib import Path
from typing import List, Dict
import random
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"Error: Could not import vllm: {e}")
    print("Make sure vllm is installed: pip install vllm")
    print("If you see CUDA errors, try: pip install vllm --upgrade")
    exit(1)


def load_config(config_file: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_prompt_template(template_file: str = "prompts/gen_from_seed.md") -> str:
    """Load prompt template from markdown file"""
    with open(template_file, 'r') as f:
        return f.read()


def load_seeds(seed_file: str) -> List[Dict]:
    """Load seed questions from JSONL file"""
    seeds = []
    with open(seed_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def prepare_prompt(template: str, seed_samples: List[Dict]) -> str:
    """
    Prepare the prompt by filling in the template with sample questions
    
    Args:
        template: The prompt template with {SAMPLE_QUESTIONS} placeholder
        seed_samples: List of 2-3 seed questions to use as examples
    
    Returns:
        The filled-in prompt ready for LLM
    """
    sample_text = ""
    for i, seed in enumerate(seed_samples, 1):
        sample_text += f"[Question {i}]: {seed['instruction']}\n\n"
    
    prompt = template.replace("{SAMPLE_QUESTIONS}", sample_text)
    return prompt


def extract_instruction(text: str) -> str:
    """
    Extract the clean instruction from LLM output.
    Remove any prefixes, metadata, and instruction artifacts.
    """
    import re
    
    instruction = text.strip()
    
    # Remove instruction markers and meta-instructions that shouldn't be in output
    # These are instructions the LLM accidentally included
    patterns = [
        r'^-\s*Do\s+NOT\s+.*?\n',  # Remove "- Do NOT..." lines
        r'^-\s*Do\s+not\s+.*?\n',  # Remove "- Do not..." lines
        r'^-\s*Match\s+.*?\n',     # Remove "- Match..." lines
        r'^-\s*Ensure\s+.*?\n',    # Remove "- Ensure..." lines
        r'^-\s*Use\s+.*?\n',       # Remove "- Use..." lines
        r'^-\s*.*?\n',             # Remove any other "- ..." lines
        r'^Output\s+only\s+.*?\n', # Remove output format instructions
        r'^\*\*Output\s+.*?\*\*\n',  # Remove bold output instructions
        r'^<\|.*?\|>.*?\n',        # Remove format markers
    ]
    
    for pattern in patterns:
        instruction = re.sub(pattern, '', instruction, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove common question numbering
    instruction = re.sub(r'^\[?Question\s*\d*\]?:?\s*', '', instruction, flags=re.IGNORECASE)
    instruction = re.sub(r'^#+\s*', '', instruction)
    
    # Clean up extra whitespace
    instruction = re.sub(r'\n\s*\n+', '\n', instruction)  # Remove multiple blank lines
    instruction = instruction.strip()
    
    return instruction


def generate_questions(
    seed_file: str,
    num_questions: int,
    config: Dict,
    output_file: str = "results/generated_questions.jsonl"
) -> int:
    """
    Generate new questions from seed file using LLM
    
    Args:
        seed_file: Path to seed JSONL file
        num_questions: Number of questions to generate
        config: Configuration dictionary
        output_file: Where to save generated questions
    
    Returns:
        Number of questions generated
    """
    
    # Create output directory
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading seeds from {seed_file}...")
    seeds = load_seeds(seed_file)
    
    if not seeds:
        print(f"Error: No seeds found in {seed_file}")
        return 0
    
    print(f"✓ Loaded {len(seeds)} seed questions")
    
    # Load LLM configuration
    llm_config = config['llm']
    gen_config = config['generation']
    
    print(f"\nInitializing LLM: {llm_config['model_name']}...")
    llm = LLM(
        model=llm_config['model_name'],
        max_model_len=llm_config['max_model_len'],
        dtype=llm_config['dtype'],
        tensor_parallel_size=llm_config['tensor_parallel_size']
    )
    
    sampling_params = SamplingParams(
        temperature=llm_config['temperature'],
        top_p=llm_config['top_p'],
        max_tokens=llm_config['max_tokens']
    )
    
    print("✓ LLM initialized")
    
    # Load prompt template
    print(f"Loading prompt template...")
    template = load_prompt_template()
    
    # Extract category from seed file name
    category = Path(seed_file).stem.replace("_questions", "")
    
    generated_questions = []
    prompts_to_process = []
    seed_info_list = []
    
    print(f"\nGenerating {num_questions} questions...")
    
    for i in range(num_questions):
        # Pick 2-3 random seed questions
        seed_batch_size = gen_config['seed_batch_size']
        sample_seeds = random.sample(seeds, min(seed_batch_size, len(seeds)))
        
        # Prepare prompt
        prompt = prepare_prompt(template, sample_seeds)
        prompts_to_process.append(prompt)
        seed_info_list.append([s['uid'] for s in sample_seeds])
        
        # Generate in batches
        batch_size = gen_config['batch_size']
        if len(prompts_to_process) >= batch_size or i == num_questions - 1:
            print(f"  Processing batch ({len(prompts_to_process)} prompts)...")
            
            # Call LLM
            outputs = llm.generate(prompts_to_process, sampling_params)
            
            # Extract and save results
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                instruction = extract_instruction(generated_text)
                
                if instruction:  # Only save if we got a valid instruction
                    generated_questions.append({
                        "uid": f"gen_{len(generated_questions):06d}",
                        "instruction": instruction,
                        "category": category,
                        "source_seeds": seed_info_list[j],
                        "generation_method": "from_seed"
                    })
            
            prompts_to_process = []
            seed_info_list = []
    
    # Save to output file
    print(f"\nSaving {len(generated_questions)} generated questions to {output_file}...")
    with open(output_file, 'w') as f:
        for q in generated_questions:
            f.write(json.dumps(q) + '\n')
    
    print(f"✓ Saved to {output_file}")
    
    return len(generated_questions)


def main():
    parser = argparse.ArgumentParser(
        description="Generate new questions from seed questions using LLM"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/generation_config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default=None,
        help="Path to seed JSONL file (overrides config)"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=None,
        help="Number of questions to generate (overrides config)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (overrides config)"
    )
    parser.add_argument(
        "--all_categories",
        action="store_true",
        help="Generate from all seed categories"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 2: GENERATE QUESTIONS FROM SEEDS")
    print("="*60)
    
    # Load config
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        exit(1)
    
    config = load_config(args.config)
    
    # Get parameters (args override config)
    seed_file = args.seed_file or "seeds/web_design_questions.jsonl"
    num_questions = args.num_questions or config['generation']['num_questions_per_seed']
    output_file = args.output_file or config['generation']['output_file']
    
    # Handle all_categories flag
    if args.all_categories:
        print("\nGenerating from all categories...\n")
        seed_dir = Path("seeds")
        seed_files = sorted(seed_dir.glob("*.jsonl"))
        
        if not seed_files:
            print("Error: No seed files found in seeds/")
            exit(1)
        
        total_generated = 0
        for seed_file in seed_files:
            category = seed_file.stem.replace("_questions", "")
            category_output = f"results/{category}_generated.jsonl"
            
            print(f"Generating from {category}...")
            
            # Validate seed file
            if not seed_file.exists():
                print(f"  ✗ Seed file not found: {seed_file}")
                continue
            
            # Generate questions
            count = generate_questions(str(seed_file), num_questions, config, category_output)
            total_generated += count
        
        print("\n" + "="*60)
        print(f"✓ Generated {total_generated} questions total from {len(seed_files)} categories")
        print("="*60)
    else:
        # Validate seed file
        if not Path(seed_file).exists():
            print(f"Error: Seed file not found: {seed_file}")
            exit(1)
        
        # Generate questions
        count = generate_questions(seed_file, num_questions, config, output_file)
        
        print("\n" + "="*60)
        print(f"✓ Generated {count} questions")
        print("="*60)


if __name__ == "__main__":
    main()
