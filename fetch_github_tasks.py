"""
Script to fetch high-quality coding tasks 
Saves them to collected_tasks.json

Categories:
- problem_solving
- web_design
- diagram_creation
- creative_coding
- scientific_computing
- game_development
"""

import requests
import json
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Get GitHub token from environment (optional but recommended)
# Get your token at: https://github.com/settings/tokens
# Add it to .env file: GITHUB_TOKEN=your_token_here
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)

# Set up headers with authentication
HEADERS = {
    "Accept": "application/vnd.github.v3+json"
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

# Map repositories to categories
REPO_CATEGORIES = {
    # Problem Solving (Algorithms & Data Structures)
    "problem_solving": [
        "TheAlgorithms/Python",
        "TheAlgorithms/JavaScript",
        "30-seconds/30-seconds-of-code",
        "donnemartin/system-design-primer",
        "yangshun/tech-interview-handbook",
    ],
    
    # Web Design (Frontend)
    "web_design": [
        "facebook/react",
        "vuejs/vue",
        "angular/angular",
        "sveltejs/svelte",
        "remix-run/remix",
        "nextjs/next.js",
    ],
    
    # Diagram Creation (Visualization)
    "diagram_creation": [
        "d3/d3",
        "apache/echarts",
        "plotly/plotly.js",
        "mermaid-js/mermaid",
        "cytoscape/cytoscape.js",
    ],
    
    # Creative Coding (Generative Art, Creative Projects)
    "creative_coding": [
        "processing/processing",
        "ml5js/ml5.js",
        "jnordberg/gif.js",
        "paper/paper.js",
        "jwagner/smartcrop.js",
    ],
    
    # Scientific Computing (Data Science, ML)
    "scientific_computing": [
        "numpy/numpy",
        "pandas-dev/pandas",
        "scikit-learn/scikit-learn",
        "tensorflow/tensorflow",
        "pytorch/pytorch",
    ],
    
    # Game Development
    "game_development": [
        "godotengine/godot",
        "libgdx/libgdx",
        "BabylonJS/Babylon.js",
        "PlayCanvasEngine/engine",
        "cocos2d/cocos2d-x",
    ],
}

# Labels that indicate good tasks for beginners
LABELS = [
    "good-first-issue",
    "help-wanted",
    "beginner-friendly",
    "good first issue",
    "junior-friendly",
    "documentation",
    "enhancement",
]

def fetch_github_issues(repo, labels=None, max_issues=10):
    """
    Fetch issues from a GitHub repository
    
    Args:
        repo: Repository in format "owner/repo"
        labels: List of labels to filter by
        max_issues: Maximum issues to fetch per label
    
    Returns:
        List of issue dictionaries
    """
    if labels is None:
        labels = LABELS
    
    issues = []
    
    for label in labels:
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "labels": label,
            "state": "open",
            "per_page": max_issues,
            "sort": "created"
        }
        
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=5)
            
            # Handle rate limiting gracefully
            if response.status_code == 403:
                remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
                limit = response.headers.get('X-RateLimit-Limit', 'unknown')
                print(f"    ⚠️  Rate limited! Remaining: {remaining}/{limit}")
                return issues  # Return what we have so far
            
            response.raise_for_status()
            
            repo_issues = response.json()
            
            for issue in repo_issues:
                # Skip pull requests
                if "pull_request" in issue:
                    continue
                
                # Skip if no body
                if not issue.get('body'):
                    continue
                
                # Limit instruction length
                instruction = f"{issue['title']}\n\n{issue.get('body', '')}"
                instruction = instruction[:400]  # Limit to 400 chars
                
                issue_data = {
                    "uid": f"github-{repo.replace('/', '-')}-{issue['number']}",
                    "instruction": instruction,
                    "category": None,  # Will be set by caller
                    "difficulty": determine_difficulty(issue),
                    "language": determine_language(repo),
                    "source": issue['html_url'],
                    "source_name": f"GitHub - {repo}",
                    "source_type": "github",
                    "author": issue['user']['login'],
                    "fetched_date": datetime.now().isoformat()
                }
                
                issues.append(issue_data)
                
                if len(issues) >= max_issues:
                    return issues
            
        except requests.exceptions.RequestException as e:
            pass  # Continue to next label
    
    return issues

def determine_category(repo, issue=None):
    """Determine task category based on repository"""
    # This is deprecated - category is now passed from caller
    return "general"

def determine_difficulty(issue):
    """Determine difficulty based on issue labels"""
    labels = [label['name'].lower() for label in issue.get('labels', [])]
    
    if any(word in ' '.join(labels) for word in ['easy', 'beginner', 'simple']):
        return "easy"
    elif any(word in ' '.join(labels) for word in ['hard', 'complex', 'advanced']):
        return "hard"
    else:
        return "medium"

def determine_language(repo):
    """Determine programming language based on repository"""
    repo_lower = repo.lower()
    
    if any(x in repo_lower for x in ['react', 'vue', 'angular', 'svelte', 'plotly', 'echarts', 'mermaid', 'cytoscape', 'd3', 'babylon', 'playcanvas']):
        return "javascript"
    elif any(x in repo_lower for x in ['python', 'numpy', 'pandas', 'sklearn', 'tensorflow', 'pytorch']):
        return "python"
    elif any(x in repo_lower for x in ['java', 'libgdx']):
        return "java"
    elif any(x in repo_lower for x in ['cpp', 'c++', 'godot', 'cocos']):
        return "cpp"
    elif any(x in repo_lower for x in ['go', 'golang']):
        return "go"
    elif any(x in repo_lower for x in ['typescript', 'ts']):
        return "typescript"
    else:
        return "general"

def main():
    """Main function to fetch and save GitHub issues"""
    
    print("=" * 80)
    print("BigCodeArena - Multi-Category Task Fetcher")
    print("=" * 80)
    # Show authentication status
    if GITHUB_TOKEN:
        print("✅ GitHub Token: AUTHENTICATED (5,000 requests/hour)")
    else:
        print("⚠️  GitHub Token: NOT PROVIDED (60 requests/hour limit)")
        print("   To use a token, run: export GITHUB_TOKEN='your_token_here'")
        print("   Get token at: https://github.com/settings/tokens")
    print()
    
    all_issues = []
    category_counts = {cat: 0 for cat in REPO_CATEGORIES.keys()}
    
    # Fetch from each category
    for category, repos in REPO_CATEGORIES.items():
        print(f"� Fetching {category.upper()} tasks...")
        
        category_issues = []
        
        for repo in repos:
            # Use higher max_issues with authentication
            max_issues = 10 if GITHUB_TOKEN else 5
            issues = fetch_github_issues(repo, max_issues=max_issues)
            
            # Set category for each issue
            for issue in issues:
                issue['category'] = category
            
            if issues:
                print(f"  ✓ {repo}: {len(issues)} tasks")
                category_issues.extend(issues)
            else:
                print(f"  • {repo}: 0 tasks")
        
        print(f"  Total for {category}: {len(category_issues)} tasks")
        print()
        
        all_issues.extend(category_issues)
        category_counts[category] = len(category_issues)
    
    # Save to file
    output_file = "autocodearena/data/collected_tasks.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_issues, f, indent=2)
    
    # Print summary
    print("=" * 80)
    print("📊 COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total tasks collected: {len(all_issues)}")
    print()
    print("By Category:")
    for category, count in category_counts.items():
        percentage = (count / len(all_issues) * 100) if all_issues else 0
        print(f"  - {category:25s}: {count:3d} tasks ({percentage:5.1f}%)")
    print()
    print(f"Output file: {output_file}")
    print("=" * 80)
    print()
    
    return all_issues

if __name__ == "__main__":
    # Check if requests library is installed
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library not found")
        print("Install it with: pip install requests")
        exit(1)
    
    issues = main()
