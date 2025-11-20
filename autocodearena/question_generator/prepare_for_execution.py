#!/usr/bin/env python3
"""
Prepare generated questions and solutions for execution visualization.
Converts solution + test cases to generation.jsonl format expected by gen_execution.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sandbox.code_analyzer import extract_code_from_markdown


def convert_solutions_to_generation_format(solutions_file: str, output_dir: str) -> int:
    """
    Convert solutions_and_tests.jsonl to generation.jsonl format for execution.
    
    Args:
        solutions_file: Path to solutions_and_tests.jsonl or solutions_and_tests_pretty.json
        output_dir: Directory where generation.jsonl will be created
        
    Returns:
        Number of records converted
    """
    
    # Load solutions
    solutions = []
    
    if solutions_file.endswith('.json'):
        with open(solutions_file) as f:
            solutions = json.load(f)
    else:  # .jsonl
        with open(solutions_file) as f:
            for line in f:
                if line.strip():
                    solutions.append(json.loads(line))
    
    print(f"📖 Loaded {len(solutions)} solutions from {os.path.basename(solutions_file)}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    generation_file = os.path.join(output_dir, "generation.jsonl")
    
    # Convert each solution to generation format
    converted_count = 0
    skipped = []
    
    with open(generation_file, 'w') as f:
        for solution in solutions:
            try:
                uid = solution.get('uid', '')
                category = solution.get('category', '')
                instruction = solution.get('instruction', '')
                code = solution.get('solution', '')
                
                if not code or not uid:
                    skipped.append(uid or 'unknown')
                    continue
                
                # Extract code and determine environment
                extraction_result = extract_code_from_markdown(code)
                
                if extraction_result:
                    code_extracted, language, dependencies, environment = extraction_result
                else:
                    # Fallback: manually detect language and environment
                    code_extracted = code
                    language = 'unknown'
                    dependencies = ([], [])
                    
                    # Detect environment based on code content - check more carefully
                    code_lower = code_extracted.lower()
                    
                    # HTML detection (highest priority)
                    if code_extracted.strip().startswith(('<!doctype', '<html', '<!DOCTYPE', '<HTML')):
                        environment = 'HTML'
                        language = 'html'
                    # CSS/HTML with styles
                    elif 'height: 100vh' in code or 'width: 100%' in code or '<style>' in code:
                        environment = 'HTML'
                        language = 'html'
                    # React detection
                    elif 'import React' in code or 'from react' in code_lower or 'import * as React' in code:
                        environment = 'React'
                        language = 'javascript'
                    # Streamlit detection
                    elif 'import streamlit' in code_lower:
                        environment = 'Streamlit'
                        language = 'python'
                    # PyGame detection
                    elif 'import pygame' in code_lower:
                        environment = 'PyGame'
                        language = 'python'
                    # Vue detection
                    elif 'import vue' in code_lower or 'from vue' in code_lower or '<template>' in code:
                        environment = 'Vue'
                        language = 'javascript'
                    # Gradio detection
                    elif 'import gradio' in code_lower:
                        environment = 'Gradio'
                        language = 'python'
                    # Mermaid detection
                    elif 'mermaid' in code_lower and ('graph' in code_lower or 'flowchart' in code_lower):
                        environment = 'Mermaid'
                        language = 'mermaid'
                    # Three.js detection
                    elif 'three' in code_lower or 'THREE' in code or 'OrbitControls' in code:
                        environment = 'React'
                        language = 'javascript'
                    # SVG/Canvas detection
                    elif '<svg' in code_lower or 'canvas' in code_lower:
                        environment = 'HTML'
                        language = 'html'
                    # JavaScript with comments (has // or /*)
                    elif '//' in code or '/*' in code:
                        # But check if it's not Python with URLs
                        if 'def ' in code or 'class ' in code or 'import ' in code:
                            environment = 'Python Runner'
                        else:
                            environment = 'React'
                            language = 'javascript'
                    else:
                        environment = 'Python Runner'  # Default fallback
                
                # Convert to generation format
                generation_record = {
                    'uid': uid,
                    'category': category,
                    'instruction': instruction,
                    'code_to_execute': code_extracted,
                    'code_dependencies': dependencies,
                    'language': language,
                    'environment': str(environment) if environment else 'Python Runner',
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"Question: {instruction}"
                        },
                        {
                            'role': 'assistant',
                            'content': {
                                'answer': code
                            }
                        }
                    ]
                }
                
                f.write(json.dumps(generation_record, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # Print sample for verification
                if converted_count <= 3:
                    print(f"  ✓ {uid}: {environment}")
                
            except Exception as e:
                print(f"  ❌ Error converting {uid}: {str(e)}")
                skipped.append(uid or 'unknown')
                continue
    
    print(f"\n✅ Converted {converted_count} solutions to generation.jsonl")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} records")
    
    print(f"📁 Output: {generation_file}")
    return converted_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare solutions for execution visualization"
    )
    parser.add_argument(
        "--solutions",
        default="solutions/solutions_and_tests_pretty.json",
        help="Path to solutions file (JSON or JSONL)"
    )
    parser.add_argument(
        "--output",
        default="../data/autocodearena_local/model_answer/generated_questions",
        help="Output directory for generation.jsonl"
    )
    
    args = parser.parse_args()
    
    # Resolve paths from current directory
    solutions_path = Path(args.solutions).resolve()
    output_path = Path(args.output).resolve()
    
    if not solutions_path.exists():
        print(f"❌ Solutions file not found: {solutions_path}")
        return
    
    print(f"🔄 Converting solutions to generation.jsonl format...")
    print(f"   Input: {solutions_path}")
    
    converted = convert_solutions_to_generation_format(
        str(solutions_path),
        str(output_path)
    )
    
    if converted > 0:
        print(f"\n🚀 Ready to run execution:")
        print(f"   cd /home/nsl/madhumini/bigcodearena/autocodearena")
        print(f"   python gen_execution.py --model_name generated_questions --max_workers 5")


if __name__ == "__main__":
    main()
