#!/usr/bin/env python3
"""
Local Judge using vLLM with open-source models (Mistral/Llama)
Evaluates code quality without needing OpenAI API
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import requests
import time

class LocalJudge:
    """Judge code quality using local vLLM instance."""
    
    def __init__(self, vllm_url="http://localhost:8000", model_name="default"):
        """
        Initialize local judge.
        
        Args:
            vllm_url: URL of vLLM server (default: http://localhost:8000)
            model_name: Name of model being judged (for logging)
        """
        self.vllm_url = vllm_url.rstrip('/')
        self.model_name = model_name
        self.api_endpoint = f"{self.vllm_url}/v1/chat/completions"
        
        # Check if vLLM server is running
        if not self._check_server():
            raise RuntimeError(f"vLLM server not running at {self.vllm_url}")
    
    def _check_server(self) -> bool:
        """Check if vLLM server is running."""
        try:
            response = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def evaluate_code(self, question: str, code: str, stdout: str = "", stderr: str = "") -> Dict:
        """
        Evaluate a code solution.
        
        Args:
            question: The original question/task
            code: The generated code
            stdout: Output from running the code
            stderr: Error output (if any)
            
        Returns:
            Dict with evaluation results
        """
        
        # Build evaluation prompt
        prompt = self._build_eval_prompt(question, code, stdout, stderr)
        
        # Call vLLM
        response = self._call_vllm(prompt)
        
        if not response:
            return {
                'status': 'error',
                'score': 0,
                'reason': 'Failed to get response from vLLM'
            }
        
        # Parse response
        evaluation = self._parse_evaluation(response)
        
        return evaluation
    
    def _build_eval_prompt(self, question: str, code: str, stdout: str, stderr: str) -> str:
        """Build evaluation prompt for the judge."""
        
        # Truncate long content
        max_len = 2000
        question_truncated = question[:max_len] if len(question) > max_len else question
        code_truncated = code[:max_len] if len(code) > max_len else code
        stdout_truncated = stdout[:500] if len(stdout) > 500 else stdout
        stderr_truncated = stderr[:500] if len(stderr) > 500 else stderr
        
        prompt = f"""You are a code quality evaluator. Evaluate the following code solution:

TASK/QUESTION:
{question_truncated}

GENERATED CODE:
```python
{code_truncated}
```

EXECUTION OUTPUT (stdout):
{stdout_truncated if stdout_truncated else "(no output)"}

EXECUTION ERROR (stderr):
{stderr_truncated if stderr_truncated else "(no errors)"}

Evaluate this code on the following criteria and provide a score from 0-100:

1. **Completeness** (0-30): Does the code fully implement the required functionality?
   - 0: Incomplete, just outline or pseudocode
   - 15: Partial implementation, missing key features
   - 30: Full implementation of all requirements

2. **Correctness** (0-30): Is the code logically correct?
   - 0: Completely wrong or non-functional
   - 15: Partially correct but has bugs
   - 30: Correct and handles edge cases

3. **Output Quality** (0-25): Does the code produce meaningful output?
   - 0: No output or error output
   - 12: Partial/unclear output
   - 25: Clear, correct, well-formatted output

4. **Code Quality** (0-15): Is the code well-written?
   - 0: Messy, unreadable
   - 7: Acceptable, readable
   - 15: Clean, well-structured

Respond in this exact JSON format:
{{
    "completeness_score": <0-30>,
    "correctness_score": <0-30>,
    "output_score": <0-25>,
    "quality_score": <0-15>,
    "total_score": <0-100>,
    "verdict": "<PASS|FAIL|PARTIAL>",
    "reasoning": "<brief explanation of verdict>"
}}

Only respond with the JSON, no other text."""
        
        return prompt
    
    def _call_vllm(self, prompt: str, max_retries: int = 3) -> str:
        """Call vLLM server with the prompt."""
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,  # Low temperature for consistent evaluation
                        "max_tokens": 500,
                        "top_p": 0.95
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        return data['choices'][0]['message']['content']
                
                print(f"Error: HTTP {response.status_code}")
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except Exception as e:
                print(f"Error calling vLLM: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None
    
    def _parse_evaluation(self, response: str) -> Dict:
        """Parse the JSON evaluation response."""
        try:
            # Extract JSON from response (in case there's extra text)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                evaluation = json.loads(json_str)
                
                # Ensure all required fields exist
                evaluation.setdefault('status', 'success')
                evaluation.setdefault('total_score', 0)
                evaluation.setdefault('verdict', 'FAIL')
                
                return evaluation
        except json.JSONDecodeError:
            pass
        
        # Fallback if JSON parsing fails
        return {
            'status': 'error',
            'score': 0,
            'reason': 'Failed to parse evaluation response'
        }


def judge_model_results(model_name: str, num_cases: int = 10, vllm_url: str = "http://localhost:8000"):
    """Judge a subset of model results."""
    
    print(f"\n{'='*60}")
    print(f"Local Judge Evaluation: {model_name}")
    print(f"{'='*60}\n")
    
    # Initialize judge
    try:
        judge = LocalJudge(vllm_url=vllm_url, model_name=model_name)
    except RuntimeError as e:
        print(f"❌ {e}")
        print("\n💡 To start vLLM, run:")
        print(f"   ./setup_vllm.sh mistral-7b-vllm 8000")
        print(f"   (or use any other config key from autocodearena/config/vllm_config.yaml)")
        return
    
    # Load data
    results_dir = Path(f"autocodearena/data/autocodearena_local/model_answer/{model_name}")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Load generation and execution data
    gen_data = {}
    with open(results_dir / "generation.jsonl") as f:
        for line in f:
            record = json.loads(line)
            gen_data[record['uid']] = record
    
    exec_results = []
    with open(results_dir / "execution_results.jsonl") as f:
        for line in f:
            exec_results.append(json.loads(line))
    
    # Prepare cases to evaluate (first num_cases)
    cases_to_eval = exec_results[:num_cases]
    
    results = []
    passes = 0
    partials = 0
    fails = 0
    
    for i, result in enumerate(cases_to_eval, 1):
        uid = result['uid']
        gen = gen_data.get(uid, {})
        
        question = gen.get('instruction', 'N/A')
        code = ''
        
        # Extract code
        if gen.get('messages'):
            for msg in gen['messages']:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', {})
                    if isinstance(content, dict):
                        code = content.get('answer', '')
                    break
        
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        category = result.get('category', 'Unknown')
        
        print(f"[{i}/{len(cases_to_eval)}] Evaluating: {category}...", end=' ', flush=True)
        
        # Evaluate
        evaluation = judge.evaluate_code(question, code, stdout, stderr)
        
        verdict = evaluation.get('verdict', 'FAIL')
        score = evaluation.get('total_score', 0)
        
        # Count results
        if verdict == 'PASS':
            passes += 1
            status_icon = "✅"
        elif verdict == 'PARTIAL':
            partials += 1
            status_icon = "⚠️"
        else:
            fails += 1
            status_icon = "❌"
        
        print(f"{status_icon} Score: {score}/100")
        
        results.append({
            'uid': uid,
            'category': category,
            'verdict': verdict,
            'score': score,
            'evaluation': evaluation
        })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed:  {passes}/{len(cases_to_eval)}")
    print(f"⚠️  Partial: {partials}/{len(cases_to_eval)}")
    print(f"❌ Failed:  {fails}/{len(cases_to_eval)}")
    print(f"Average Score: {sum(r['score'] for r in results) / len(results):.1f}/100")
    print(f"{'='*60}\n")
    
    # Save results
    output_file = f"judge_results_{model_name}.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"✅ Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python local_judge.py <model_name> [num_cases] [vllm_url]")
        print("\nExample:")
        print("  python local_judge.py qwen3-4b-inst-2507-vllm 10 http://localhost:8000")
        print("  python local_judge.py phi-2-vllm 5")
        sys.exit(1)
    
    model_name = sys.argv[1]
    num_cases = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    vllm_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    
    judge_model_results(model_name, num_cases, vllm_url)
