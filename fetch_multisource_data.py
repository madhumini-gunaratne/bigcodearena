"""
Fetch data from multiple high-quality sources:
1. Stack Overflow - Top answers by votes for keywords
2. Official Documentation - Curated official docs by category/language
3. GitHub Code Examples - From well-maintained repos (not issues)

Saves to: autocodearena/data/new data v1.1/collected_multisource_data.json
"""

import requests
import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from urllib.parse import urljoin

# ============================================================================
# CONFIGURATION
# ============================================================================

# High-quality GitHub repos with good examples and documentation
QUALITY_REPOS = {
    "problem_solving": [
        "TheAlgorithms/Python",
        "TheAlgorithms/JavaScript",
        "trekhleb/javascript-algorithms",
    ],
    "web_design": [
        "facebook/react",
        "vuejs/vue",
        "angular/angular",
    ],
    "diagram_creation": [
        "d3/d3",
        "apache/echarts",
        "plotly/plotly.js",
    ],
    "creative_coding": [
        "processing/processing",
        "ml5js/ml5.js",
        "paper/paper.js",
    ],
    "scientific_computing": [
        "numpy/numpy",
        "pandas-dev/pandas",
        "scikit-learn/scikit-learn",
    ],
    "game_development": [
        "godotengine/godot",
        "BabylonJS/Babylon.js",
    ],
}

# Curated official documentation by category and language
OFFICIAL_DOCS = {
    "problem_solving": {
        "python": [
            {
                "title": "Python Data Structures - Official Docs",
                "url": "https://docs.python.org/3/tutorial/datastructures.html",
                "topics": ["list", "dict", "set", "tuple", "array"],
                "language": "python",
            },
            {
                "title": "Python Algorithm Design Manual",
                "url": "https://docs.python.org/3/library/heapq.html",
                "topics": ["heap", "sort", "algorithm"],
                "language": "python",
            },
        ],
        "javascript": [
            {
                "title": "MDN - Array Methods",
                "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array",
                "topics": ["array", "algorithm", "sort"],
                "language": "javascript",
            },
            {
                "title": "MDN - Object Basics",
                "url": "https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects",
                "topics": ["object", "data structure"],
                "language": "javascript",
            },
        ],
    },
    "web_design": {
        "javascript": [
            {
                "title": "React Official Documentation",
                "url": "https://react.dev/",
                "topics": ["react", "component", "hooks", "state"],
                "language": "javascript",
            },
            {
                "title": "MDN Web Docs - HTML/CSS/JavaScript",
                "url": "https://developer.mozilla.org/en-US/",
                "topics": ["html", "css", "dom", "web"],
                "language": "javascript",
            },
            {
                "title": "Vue.js Official Guide",
                "url": "https://vuejs.org/guide/",
                "topics": ["vue", "component", "state"],
                "language": "javascript",
            },
            {
                "title": "CSS-Tricks",
                "url": "https://css-tricks.com/",
                "topics": ["css", "layout", "design"],
                "language": "javascript",
            },
        ],
    },
    "diagram_creation": {
        "javascript": [
            {
                "title": "D3.js Official Documentation",
                "url": "https://d3js.org/",
                "topics": ["d3", "visualization", "data", "chart"],
                "language": "javascript",
            },
            {
                "title": "Plotly.js Documentation",
                "url": "https://plotly.com/javascript/",
                "topics": ["plotly", "chart", "visualization"],
                "language": "javascript",
            },
        ],
        "python": [
            {
                "title": "Matplotlib Documentation",
                "url": "https://matplotlib.org/stable/contents.html",
                "topics": ["matplotlib", "visualization", "plot"],
                "language": "python",
            },
        ],
    },
    "creative_coding": {
        "javascript": [
            {
                "title": "p5.js Reference",
                "url": "https://p5js.org/reference/",
                "topics": ["p5", "creative", "animation", "graphics"],
                "language": "javascript",
            },
            {
                "title": "Three.js Documentation",
                "url": "https://threejs.org/docs/",
                "topics": ["three.js", "3d", "graphics"],
                "language": "javascript",
            },
        ],
    },
    "scientific_computing": {
        "python": [
            {
                "title": "NumPy Documentation",
                "url": "https://numpy.org/doc/",
                "topics": ["numpy", "array", "scientific"],
                "language": "python",
            },
            {
                "title": "Pandas Documentation",
                "url": "https://pandas.pydata.org/docs/",
                "topics": ["pandas", "dataframe", "data analysis"],
                "language": "python",
            },
            {
                "title": "Scikit-learn Documentation",
                "url": "https://scikit-learn.org/stable/documentation.html",
                "topics": ["sklearn", "machine learning", "classification"],
                "language": "python",
            },
        ],
    },
    "game_development": {
        "javascript": [
            {
                "title": "Babylon.js Documentation",
                "url": "https://www.babylonjs-playground.com/",
                "topics": ["babylon.js", "3d", "game"],
                "language": "javascript",
            },
        ],
    },
}

# Technical keywords for extraction
TECHNICAL_KEYWORDS = {
    "tree", "array", "list", "sort", "search", "graph", "algorithm",
    "recursive", "cache", "dynamic", "binary", "hash", "queue", "stack",
    "react", "component", "state", "hook", "api", "async", "promise",
    "visualization", "chart", "d3", "animation", "3d", "game", "mesh",
    "dataframe", "numpy", "sklearn", "neural", "network", "model",
    "websocket", "event", "listener", "dom", "css", "html",
    "optimization", "performance", "memory", "time", "complexity"
}

# Stack Overflow tag mappings for categories
SO_TAG_MAPPINGS = {
    "problem_solving": ["algorithm", "data-structures", "sorting", "recursion"],
    "web_design": ["javascript", "react", "html", "css", "angular", "vue.js"],
    "diagram_creation": ["d3.js", "visualization", "charts", "plotly"],
    "creative_coding": ["animation", "p5.js", "three.js", "generative-art"],
    "scientific_computing": ["numpy", "pandas", "scikit-learn", "machine-learning"],
    "game_development": ["game-development", "babylon.js", "unity3d"],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_keywords(instruction):
    """Extract technical keywords from instruction text"""
    words = re.findall(r'\b\w+\b', instruction.lower())
    keywords = [w for w in set(words) if w in TECHNICAL_KEYWORDS]
    return keywords


def detect_language(instruction):
    """Detect programming language from instruction"""
    instruction_lower = instruction.lower()
    
    language_indicators = {
        "python": ["python", "pip", "pandas", "numpy", "django", "flask"],
        "javascript": ["javascript", "js", "node", "react", "vue", "npm"],
        "java": ["java", "maven", "spring"],
        "cpp": ["c++", "cpp", "template"],
    }
    
    for lang, indicators in language_indicators.items():
        if any(ind in instruction_lower for ind in indicators):
            return lang
    
    # Default based on content
    if any(w in instruction_lower for w in ["html", "css", "react", "dom", "browser"]):
        return "javascript"
    return "python"


# ============================================================================
# STACK OVERFLOW DATA COLLECTION
# ============================================================================

def fetch_stackoverflow_data(keywords, limit=5):
    """
    Fetch top Stack Overflow questions/answers for given keywords
    
    Args:
        keywords: list of technical keywords to search
        limit: number of results to return
    
    Returns:
        list of dicts with SO Q&A data
    """
    if not keywords:
        return []
    
    results = []
    
    # Try searching by tags (more reliable)
    tags = ";".join(keywords[:3])
    
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "tagged": tags,
        "sort": "votes",
        "order": "desc",
        "site": "stackoverflow",
        "pagesize": limit,
        "filter": "withbody",
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            items = response.json().get("items", [])
            
            for item in items:
                results.append({
                    "uid": f"so-{item['question_id']}",
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "votes": item.get("score", 0),
                    "answer_count": item.get("answer_count", 0),
                    "tags": item.get("tags", []),
                    "body_excerpt": item.get("body", "")[:200],
                    "source": "stackoverflow",
                    "source_type": "stackoverflow",
                    "fetched_date": datetime.now().isoformat(),
                })
        
        # Rate limiting - be respectful
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  ⚠️  Error fetching from Stack Overflow: {e}")
    
    return results[:limit]


# ============================================================================
# GITHUB REPO CODE EXAMPLES
# ============================================================================

def fetch_github_repo_examples(repo, category, limit=3):
    """
    Fetch code example links from a GitHub repo
    
    Returns links to:
    - README examples
    - Example folders
    - Well-documented test files
    
    Args:
        repo: repo in format "owner/repo"
        category: category for context
        limit: max examples to return
    
    Returns:
        list of example references
    """
    examples = []
    
    try:
        # Try to get README
        url = f"https://api.github.com/repos/{repo}/readme"
        headers = {"Accept": "application/vnd.github.v3.raw"}
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            readme = response.text
            
            # Extract example links from README
            # Look for common patterns: examples/, docs/, tutorial links
            example_patterns = [
                r'https://github\.com/[^/]+/[^/]+/blob/[^\s)]+',  # Direct links
                r'\[.+?\]\((examples?/[^\)]+)\)',  # Markdown links to examples
            ]
            
            for pattern in example_patterns:
                matches = re.findall(pattern, readme)
                examples.extend(matches[:2])
        
        # Also add direct links to common example locations
        repo_base = f"https://github.com/{repo}"
        example_urls = [
            f"{repo_base}/tree/main/examples",
            f"{repo_base}/tree/main/demo",
            f"{repo_base}/tree/main/samples",
        ]
        
        for url in example_urls:
            examples.append(url)
        
        time.sleep(0.3)  # Rate limiting
        
    except Exception as e:
        print(f"  ⚠️  Error fetching examples from {repo}: {e}")
    
    return [
        {
            "uid": f"github-{repo.replace('/', '-')}-example",
            "title": f"Code Examples - {repo}",
            "url": url,
            "repo": repo,
            "source": "github_repo",
            "source_type": "github_repo",
            "fetched_date": datetime.now().isoformat(),
        }
        for url in examples[:limit]
    ]


# ============================================================================
# OFFICIAL DOCUMENTATION COLLECTION
# ============================================================================

def get_official_docs(category, language=None):
    """Get curated official documentation for category"""
    if category not in OFFICIAL_DOCS:
        return []
    
    docs = []
    category_docs = OFFICIAL_DOCS[category]
    
    if language:
        # Get docs for specific language
        if language in category_docs:
            for doc in category_docs[language]:
                docs.append({
                    "uid": f"docs-{language}-{doc['title'].lower().replace(' ', '-')}",
                    "title": doc["title"],
                    "url": doc["url"],
                    "topics": doc.get("topics", []),
                    "language": doc.get("language", language),
                    "source": "official_docs",
                    "source_type": "official_docs",
                    "fetched_date": datetime.now().isoformat(),
                })
    else:
        # Get all docs for category
        for lang, doc_list in category_docs.items():
            for doc in doc_list:
                docs.append({
                    "uid": f"docs-{lang}-{doc['title'].lower().replace(' ', '-')}",
                    "title": doc["title"],
                    "url": doc["url"],
                    "topics": doc.get("topics", []),
                    "language": doc.get("language", lang),
                    "source": "official_docs",
                    "source_type": "official_docs",
                    "fetched_date": datetime.now().isoformat(),
                })
    
    return docs


# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

def collect_multisource_data():
    """
    Collect data from all sources:
    1. Stack Overflow (by keywords)
    2. Official Documentation (by category/language)
    3. GitHub Repo Examples (by category)
    """
    
    print("=" * 80)
    print("🚀 Collecting Multi-Source Enrichment Data")
    print("=" * 80)
    print()
    
    all_data = []
    stats = defaultdict(int)
    
    # ====== STACK OVERFLOW ======
    print("📚 Fetching from Stack Overflow...")
    print()
    
    so_data = []
    for keyword in list(TECHNICAL_KEYWORDS)[:10]:  # Sample of keywords
        print(f"  🔍 Searching: {keyword}")
        results = fetch_stackoverflow_data([keyword], limit=2)
        so_data.extend(results)
        stats["stackoverflow"] += len(results)
    
    # Deduplicate by URL
    so_urls_seen = set()
    so_unique = []
    for item in so_data:
        if item["url"] not in so_urls_seen:
            so_urls_seen.add(item["url"])
            so_unique.append(item)
    
    all_data.extend(so_unique)
    print(f"  ✅ Collected {len(so_unique)} unique Stack Overflow Q&A")
    print()
    
    # ====== OFFICIAL DOCUMENTATION ======
    print("📖 Collecting Official Documentation...")
    print()
    
    docs_data = []
    for category in OFFICIAL_DOCS.keys():
        category_docs = get_official_docs(category)
        docs_data.extend(category_docs)
        print(f"  ✅ {category}: {len(category_docs)} docs")
        stats["official_docs"] += len(category_docs)
    
    all_data.extend(docs_data)
    print(f"  ✅ Total: {len(docs_data)} official documentation links")
    print()
    
    # ====== GITHUB REPO EXAMPLES ======
    print("🐙 Collecting GitHub Repository Examples...")
    print()
    
    github_data = []
    for category, repos in QUALITY_REPOS.items():
        print(f"  📁 {category}:")
        for repo in repos[:2]:  # Limit to 2 repos per category
            try:
                examples = fetch_github_repo_examples(repo, category, limit=2)
                github_data.extend(examples)
                print(f"    ✅ {repo}: {len(examples)} examples")
                stats["github_repo"] += len(examples)
            except Exception as e:
                print(f"    ⚠️  {repo}: {e}")
        print()
    
    all_data.extend(github_data)
    print(f"  ✅ Total: {len(github_data)} GitHub repo examples")
    print()
    
    # ====== SAVE DATA ======
    print("=" * 80)
    print("💾 Saving Data")
    print("=" * 80)
    print()
    
    output_file = "autocodearena/data/new data v1.1/collected_multisource_data.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to: {output_file}")
    print()
    
    # ====== STATISTICS ======
    print("=" * 80)
    print("📊 Collection Summary")
    print("=" * 80)
    print()
    print(f"Stack Overflow Q&A:        {stats['stackoverflow']} items")
    print(f"Official Documentation:    {stats['official_docs']} items")
    print(f"GitHub Repo Examples:      {stats['github_repo']} items")
    print(f"Total:                     {len(all_data)} items")
    print()
    
    # Show samples
    print("=" * 80)
    print("📋 Sample Data Items")
    print("=" * 80)
    print()
    
    if all_data:
        print("Stack Overflow Sample:")
        so_sample = [item for item in all_data if item['source'] == 'stackoverflow']
        if so_sample:
            print(f"  Title: {so_sample[0]['title']}")
            print(f"  URL: {so_sample[0]['url']}")
            print(f"  Votes: {so_sample[0]['votes']}")
        print()
        
        print("Official Docs Sample:")
        docs_sample = [item for item in all_data if item['source'] == 'official_docs']
        if docs_sample:
            print(f"  Title: {docs_sample[0]['title']}")
            print(f"  URL: {docs_sample[0]['url']}")
            print(f"  Topics: {docs_sample[0].get('topics', [])}")
        print()
        
        print("GitHub Repo Sample:")
        gh_sample = [item for item in all_data if item['source'] == 'github_repo']
        if gh_sample:
            print(f"  Title: {gh_sample[0]['title']}")
            print(f"  URL: {gh_sample[0]['url']}")
            print(f"  Repo: {gh_sample[0]['repo']}")
    
    print()
    print("=" * 80)
    print("✨ Data collection complete!")
    print("=" * 80)
    
    return all_data


if __name__ == "__main__":
    try:
        collect_multisource_data()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
