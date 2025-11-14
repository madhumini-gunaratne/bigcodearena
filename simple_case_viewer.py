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

def create_simple_viewer(model_name, num_success=2, num_error=2, target_uid=None, success_uid=None, error_uid=None, data_path=None, output_file_name=None, title=None):
    """Generate simple HTML viewer with successful and error cases, or specific UIDs."""
    
    if data_path is None:
        data_path = "autocodearena/data/autocodearena_local/model_answer"
    
    # Use custom title if provided, otherwise use model_name
    display_title = title if title else model_name
    
    results_dir = Path(f"{data_path}/{model_name}")
    
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
    
    # Normalize empty strings to None for filtering
    success_uid = success_uid if success_uid else None
    error_uid = error_uid if error_uid else None
    target_uid = target_uid if target_uid else None
    
    for r in exec_results:
        uid = r['uid']
        gen = gen_data.get(uid, {})
        
        # Skip if specific UIDs are provided and this doesn't match
        if (success_uid or error_uid or target_uid):
            # We're filtering for specific UIDs
            if not (uid == success_uid or uid == error_uid or uid == target_uid):
                continue  # Skip this one
        
        # Extract question and strip enrichment context if present
        question = gen.get('instruction', 'N/A')
        if "---\n\n**Related Real-World Examples from GitHub:**" in question:
            # Strip enrichment context for cleaner display
            question = question.split("---\n\n**Related Real-World Examples from GitHub:**")[0].strip()
        
        case = {
            'uid': uid,
            'category': r.get('category', 'Unknown'),
            'environment': r.get('environment', 'Unknown'),
            'question': question,
            'stdout': r.get('stdout', ''),
            'stderr': r.get('stderr', ''),
            'code': '',
            'image_base64': None,
            'has_error': bool(r.get('stderr')),
            'note': '',
            'is_false_positive': uid == error_uid  # Mark if this is the false positive case
        }
        
        # Add note for false positive cases
        if case['is_false_positive']:
            case['note'] = '⚠️ FALSE POSITIVE: This case passes Python execution with no stderr, but the generated code contains runtime errors in the browser/JavaScript execution. The system incorrectly classified this as successful because it only checks for Python errors, not JavaScript/browser errors.'
        # Add note for other cases that pass but don't generate expected output
        elif not case['has_error'] and r.get('screenshot_path'):
            screenshot_path = results_dir / r['screenshot_path']
            if screenshot_path.exists():
                # Check if screenshot shows errors (this is a heuristic)
                # For now, we'll add a note if there's no stdout and it has a screenshot
                if not case['stdout']:
                    case['note'] = '⚠️ Note: This case passes Python execution but may contain runtime errors in the generated output. No Python stderr was captured, but the browser/runtime execution may have encountered issues.'
        
        # Extract code
        if gen.get('messages'):
            for msg in gen['messages']:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    full_answer = ''
                    if isinstance(content, dict) and 'answer' in content:
                        full_answer = content['answer']
                    else:
                        full_answer = str(content)
                    
                    # Try to extract code blocks (between ``` markers)
                    import re
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', full_answer, re.DOTALL)
                    if code_blocks:
                        # Combine all code blocks
                        case['code'] = '\n\n'.join(code_blocks)
                    else:
                        # If no code blocks, use the full answer
                        case['code'] = full_answer
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
    
    # Handle specific UIDs - override the successful/error classification
    if success_uid or error_uid:
        # Collect the requested UIDs
        all_uids_to_show = []
        if success_uid:
            all_uids_to_show.append((success_uid, 'success'))
        if error_uid:
            all_uids_to_show.append((error_uid, 'error'))
        
        # Rebuild successful and error_cases based on requested UIDs
        all_cases_by_uid = {c['uid']: c for c in successful + error_cases}
        
        successful = []
        error_cases = []
        for uid, intended_type in all_uids_to_show:
            if uid in all_cases_by_uid:
                case = all_cases_by_uid[uid]
                if intended_type == 'success':
                    successful.append(case)
                else:
                    error_cases.append(case)
    elif target_uid:
        # If target_uid is specified, only use that case
        pass
    else:
        successful = successful[:num_success]
        error_cases = error_cases[:num_error]
    all_cases = successful + error_cases
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Case Study - {display_title}</title>
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
    <h1>{display_title}</h1>
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
        {f'<div style="background: #fff8dc; padding: 12px; margin: 10px 0; border-left: 4px solid #ff9800; color: #333;">{case["note"]}</div>' if case.get('note') else ''}
        
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
    
    if output_file_name is None:
        output_file_name = f"case_study_{model_name}.html"
    
    output_file = Path(output_file_name)
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ Created: {output_file}")
    print(f"   Successful cases: {len(successful)}")
    print(f"   Error cases: {len(error_cases)}")

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-4b-inst-2507-vllm"
    num_success = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    num_error = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    target_uid = sys.argv[4] if len(sys.argv) > 4 else None
    success_uid = sys.argv[5] if len(sys.argv) > 5 else None
    error_uid = sys.argv[6] if len(sys.argv) > 6 else None
    data_path = sys.argv[7] if len(sys.argv) > 7 else None
    output_file_name = sys.argv[8] if len(sys.argv) > 8 else None
    title = sys.argv[9] if len(sys.argv) > 9 else None
    create_simple_viewer(model, num_success, num_error, target_uid, success_uid, error_uid, data_path, output_file_name, title)
