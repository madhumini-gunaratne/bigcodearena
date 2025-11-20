#!/usr/bin/env python3
"""
Quality assessment script to evaluate generated coding questions.
Uses LLM to assess questions across 6 dimensions.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import yaml
from vllm import LLM, SamplingParams
import xml.etree.ElementTree as ET
from tqdm import tqdm


def load_config():
    """Load configuration from generation_config.yaml"""
    config_path = Path(__file__).parent / "config" / "generation_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_quality_template():
    """Load quality assessment template"""
    template_path = Path(__file__).parent / "prompts" / "quality_assessment.md"
    with open(template_path) as f:
        return f.read()


def load_questions_from_file(file_path: Path) -> List[Dict]:
    """Load questions from JSONL file"""
    questions = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def prepare_assessment_prompt(template: str, question: Dict, language: str = "Python") -> str:
    """
    Prepare assessment prompt by filling template with question data.
    For generated questions without solutions/tests, focus on question quality and difficulty.
    """
    # Simplified prompt for generated questions - focus on question clarity and difficulty only
    simplified_prompt = f"""Assess this coding question on two dimensions:

1. **Question Quality** (very poor, poor, average, good, excellent): Is the question clear, well-written, and properly specified?
2. **Question Difficulty** (very easy, easy, medium, hard, very hard): What is the complexity level?

Question:
{question.get("instruction", "")}

Provide your response in XML format:
<response>
  <question_quality>
    <reasoning>Brief explanation of clarity and specification quality</reasoning>
    <rating><!-- excellent, good, average, poor, or very poor --></rating>
  </question_quality>
  <question_difficulty>
    <reasoning>Brief explanation of complexity level</reasoning>
    <rating><!-- very easy, easy, medium, hard, or very hard --></rating>
  </question_difficulty>
</response>"""
    
    return simplified_prompt


def parse_assessment_response(response_text: str) -> Dict:
    """Parse XML response from LLM assessment"""
    assessment = {
        "question_quality": None,
        "question_difficulty": None,
    }
    
    try:
        # Find XML content
        start_idx = response_text.find("<response>")
        end_idx = response_text.rfind("</response>")
        
        if start_idx == -1 or end_idx == -1:
            return None
        
        xml_str = response_text[start_idx:end_idx+len("</response>")]
        root = ET.fromstring(xml_str)
        
        for key in assessment.keys():
            elem = root.find(key)
            if elem is not None:
                rating_elem = elem.find("rating")
                reasoning_elem = elem.find("reasoning")
                if rating_elem is not None and rating_elem.text:
                    assessment[key] = {
                        "rating": rating_elem.text.strip(),
                        "reasoning": reasoning_elem.text.strip() if reasoning_elem is not None and reasoning_elem.text else None
                    }
        
        # Only return if we got at least one rating
        if assessment["question_quality"] or assessment["question_difficulty"]:
            return assessment
        return None
        
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def assess_questions(questions: List[Dict], llm, template: str, batch_size: int = 4) -> List[Dict]:
    """Assess questions using LLM"""
    results = []
    
    # Prepare prompts
    prompts = [prepare_assessment_prompt(template, q) for q in questions]
    
    print(f"Assessing {len(questions)} questions...")
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        
        # Generate assessments
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            max_tokens=3000
        )
        
        outputs = llm.generate(batch_prompts, sampling_params)
        
        for j, output in enumerate(outputs):
            response_text = output.outputs[0].text
            assessment = parse_assessment_response(response_text)
            
            if assessment:
                results.append({
                    "uid": batch_questions[j].get("uid"),
                    "instruction": batch_questions[j].get("instruction", "")[:100] + "...",
                    "category": batch_questions[j].get("category"),
                    "assessment": assessment,
                    "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
                })
    
    return results


def save_assessment_results(results: List[Dict], output_file: Path):
    """Save assessment results to file"""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Saved {len(results)} assessment results to {output_file.name}")


def print_assessment_summary(results: List[Dict]):
    """Print summary of assessment results"""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"ASSESSMENT SUMMARY ({len(results)} questions)")
    print(f"{'='*80}")
    
    # Aggregate ratings by dimension
    dimensions = [
        "question_quality",
        "question_difficulty",
    ]
    
    for dim in dimensions:
        ratings = {}
        for result in results:
            if result.get("assessment") and result["assessment"].get(dim):
                rating = result["assessment"][dim].get("rating")
                if rating:
                    ratings[rating] = ratings.get(rating, 0) + 1
        
        if ratings:
            print(f"\n{dim.upper().replace('_', ' ')}:")
            for rating in ["excellent", "good", "average", "poor", "very poor", "very easy", "easy", "medium", "hard", "very hard"]:
                if rating in ratings:
                    print(f"  {rating}: {ratings[rating]}")
    
    # Print problematic questions
    print(f"\n{'='*80}")
    print("QUESTIONS NEEDING ATTENTION:")
    print(f"{'='*80}")
    
    for result in results:
        assessment = result.get("assessment")
        if assessment:
            poor_ratings = [
                dim for dim in dimensions
                if assessment.get(dim) and assessment[dim].get("rating") in ["very poor", "poor"]
            ]
            
            if poor_ratings:
                print(f"\n[{result['category']}] {result['uid']}")
                print(f"  Issues: {', '.join(poor_ratings)}")
                print(f"  Question: {result['instruction']}")


def main():
    parser = argparse.ArgumentParser(description="Assess quality of generated coding questions")
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).parent / "results",
                        help="Directory containing deduplicated JSONL files")
    parser.add_argument("--output_file", type=Path, default=Path(__file__).parent / "assessment_results.jsonl",
                        help="Output file for assessment results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for LLM inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to assess (for testing)")
    
    args = parser.parse_args()
    
    config = load_config()
    template = load_quality_template()
    
    # Load questions from deduplicated files
    print(f"Loading deduplicated questions from {args.results_dir}...")
    all_questions = []
    
    for jsonl_file in sorted(args.results_dir.glob("*_generated_deduped.jsonl")):
        questions = load_questions_from_file(jsonl_file)
        all_questions.extend(questions)
    
    print(f"Loaded {len(all_questions)} questions")
    
    if args.limit:
        all_questions = all_questions[:args.limit]
        print(f"Limited to {len(all_questions)} questions for testing")
    
    # Initialize LLM
    model_name = config["llm"]["model_name"]
    max_model_len = config["llm"].get("max_model_len", 4096)
    print(f"\nLoading LLM: {model_name}...")
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=1
    )
    
    # Assess questions
    results = assess_questions(all_questions, llm, template, batch_size=args.batch_size)
    
    # Save results
    save_assessment_results(results, args.output_file)
    
    # Print summary
    print_assessment_summary(results)


if __name__ == "__main__":
    main()
