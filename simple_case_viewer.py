#!/usr/bin/env python3
"""
Simple case study viewer - shows question, answer, and screenshot/visualization
"""

import json
import base64
import html as html_module
from pathlib import Path
import re

def format_question(question):
    """Format question with better structure and line breaks."""
    # Escape HTML first
    q = html_module.escape(question)
    
    # Add line breaks before common keywords that indicate new sections
    q = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '<br><br>', q)
    
    # Break up long runs by inserting line breaks at logical points
    # Split on common function/method indicators
    q = re.sub(r'(\))\s+(\w+\()', r'\1<br><br>\2', q)
    
    # Format examples with ➞
    q = re.sub(r'<br><br>(\w+\([^)]*\).*?➞)', r'<br><br><strong>Example:</strong> \1', q)
    
    # Break very long lines - split at commas and 'and'
    lines = q.split('<br>')
    formatted_lines = []
    for line in lines:
        if len(line) > 150:
            # Add breaks at commas and 'and'
            line = re.sub(r',\s+', ',<br>&nbsp;&nbsp;&nbsp;&nbsp;', line)
            line = re.sub(r'\s+and\s+', ' and<br>&nbsp;&nbsp;&nbsp;&nbsp;', line)
        formatted_lines.append(line)
    
    return '<br>'.join(formatted_lines)

def create_simple_viewer(model_name, num_success=2, num_error=2):
    """Generate simple HTML viewer with successful and error cases."""
    
    results_dir = Path(f"autocodearena/data/autocodearena_local/model_answer/{model_name}")
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found")
        return
    
    # Load data
    gen_data = {}
    with open(results_dir / "generation.jsonl") as f:
        for line in f:
            record = json.loads(line)
            gen_data[record['uid']] = record
    
    exec_results = []
    with open(results_dir / "execution_results.jsonl") as f:
        for line in f:
            exec_results.append(json.loads(line))
    
    # Filter successful cases
    successful = []
    error_cases = []
    for r in exec_results:
        uid = r['uid']
        gen = gen_data.get(uid, {})
        
        case = {
            'uid': uid,
            'category': r.get('category', 'Unknown'),
            'environment': r.get('environment', 'Unknown'),
            'question': gen.get('instruction', 'N/A'),
            'stdout': r.get('stdout', ''),
            'stderr': r.get('stderr', ''),
            'code': '',
            'image_base64': None,
            'has_error': bool(r.get('stderr'))
        }
        
        # Extract code
        if gen.get('messages'):
            for msg in gen['messages']:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    if isinstance(content, dict) and 'answer' in content:
                        case['code'] = content['answer']  # Full code, no limit
                    else:
                        case['code'] = str(content)  # Full code, no limit
                    break
        
        # Get screenshot (only for successful cases)
        if not case['has_error'] and r.get('screenshot_path'):
            screenshot_path = results_dir / r['screenshot_path']
            if screenshot_path.exists():
                with open(screenshot_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                    case['image_base64'] = f"data:image/png;base64,{b64}"
        
        # Get visual output (only for successful cases)
        if not case['has_error'] and not case['image_base64'] and r.get('visual_outputs'):
            vis_path = results_dir / r['visual_outputs'][0]
            if vis_path.exists():
                with open(vis_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                    case['image_base64'] = f"data:image/png;base64,{b64}"
        
        if case['has_error']:
            error_cases.append(case)
        else:
            successful.append(case)
    
    # Limit cases
    successful = successful[:num_success]
    error_cases = error_cases[:num_error]
    all_cases = successful + error_cases
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Case Study - {model_name}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-light.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .case {{ 
            background: white; 
            margin: 20px 0; 
            padding: 20px; 
            border: 1px solid #ddd;
            max-width: 900px;
        }}
        .case.error {{ border-left: 4px solid #d9534f; }}
        .case.success {{ border-left: 4px solid #5cb85c; }}
        .case h2 {{ color: #333; margin-top: 0; }}
        .status {{ display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 0.85em; font-weight: bold; }}
        .status.success {{ background: #5cb85c; color: white; }}
        .status.error {{ background: #d9534f; color: white; }}
        .meta {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .meta span {{ margin-right: 20px; }}
        h3 {{ color: #333; margin-top: 20px; }}
        .question {{ background: #fafafa; padding: 15px; margin: 15px 0; border-left: 4px solid #333; line-height: 1.6; }}
        .question code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
        .question strong {{ color: #d9534f; }}
        .example {{ background: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 3px solid #5cb85c; border-radius: 3px; font-family: monospace; font-size: 0.85em; overflow-x: auto; }}
        .code {{ 
            background: #f5f5f5; 
            padding: 0; 
            margin: 15px 0; 
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }}
        .code pre {{ 
            margin: 0;
            padding: 15px;
            overflow-x: auto;
            font-size: 0.85em;
            line-height: 1.4;
        }}
        .image {{ margin: 15px 0; }}
        .image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .output {{ background: #f0f0f0; padding: 15px; margin: 15px 0; border: 1px solid #ddd; white-space: pre-wrap; font-family: monospace; font-size: 0.85em; }}
        .error-output {{ background: #fff3cd; padding: 15px; margin: 15px 0; border: 1px solid #ffc107; color: #856404; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>{model_name}</h1>
    <p>Successful: {len(successful)} | Errors: {len(error_cases)}</p>
"""
    
    for i, case in enumerate(successful, 1):
        visual_html = ""
        if case['image_base64']:
            visual_html = f'<div class="image"><img src="{case["image_base64"]}" alt="Result"></div>'
        else:
            visual_html = f'<div class="output"><strong>Output:</strong>\n{html_module.escape(case["stdout"][:500])}</div>'
        
        html += f"""
    <div class="case success">
        <h2>Success {i}: {case['category']} <span class="status success">SUCCESS</span></h2>
        <div class="meta">
            <span>UID: <code>{case['uid']}</code></span>
            <span>Environment: <strong>{case['environment']}</strong></span>
        </div>
        
        <h3>Question:</h3>
        <div class="question">{format_question(case['question'])}</div>
        
        <h3>Generated Code:</h3>
        <div class="code"><pre><code class="language-python">{html_module.escape(case['code'])}</code></pre></div>
        
        <h3>Result:</h3>
        {visual_html}
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
    </div>
"""
    
    for i, case in enumerate(error_cases, 1):
        html += f"""
    <div class="case error">
        <h2>Error {i}: {case['category']} <span class="status error">ERROR</span></h2>
        <div class="meta">
            <span>UID: <code>{case['uid']}</code></span>
            <span>Environment: <strong>{case['environment']}</strong></span>
        </div>
        
        <h3>Question:</h3>
        <div class="question">{format_question(case['question'])}</div>
        
        <h3>Generated Code:</h3>
        <div class="code"><pre><code class="language-python">{html_module.escape(case['code'])}</code></pre></div>
        
        <h3>Error Output:</h3>
        <div class="error-output">{html_module.escape(case['stderr'][:1000])}</div>
        
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
    </div>
"""
    
    html += """
    <script>
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    </script>
</body>
</html>
"""
    
    output_file = Path(f"case_study_{model_name}.html")
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ Created: {output_file}")
    print(f"   Successful cases: {len(successful)}")
    print(f"   Error cases: {len(error_cases)}")

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-4b-inst-2507-vllm"
    num_success = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    num_error = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    create_simple_viewer(model, num_success, num_error)
