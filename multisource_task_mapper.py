"""
Enrich tasks by mapping questions to multi-source resources
(Stack Overflow, Official Docs, GitHub examples)

Uses keyword extraction + relevance scoring to find best matches
"""

import json
import os
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

# ============================================================================
# CONFIGURATION
# ============================================================================

# Technical keywords to extract from questions
TECHNICAL_KEYWORDS = {
    "tree", "array", "list", "sort", "search", "graph", "algorithm",
    "recursive", "cache", "dynamic", "binary", "hash", "queue", "stack",
    "react", "component", "state", "hook", "api", "async", "promise",
    "visualization", "chart", "d3", "animation", "3d", "game", "mesh",
    "dataframe", "numpy", "sklearn", "neural", "network", "model",
    "websocket", "event", "listener", "dom", "css", "html",
    "optimization", "performance", "memory", "time", "complexity",
    "loop", "function", "class", "variable", "string", "number"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_keywords(instruction):
    """Extract technical keywords from instruction"""
    words = re.findall(r'\b\w+\b', instruction.lower())
    keywords = [w for w in set(words) if w in TECHNICAL_KEYWORDS]
    return list(keywords)


def detect_language(instruction):
    """Detect programming language from instruction"""
    instruction_lower = instruction.lower()
    
    language_indicators = {
        "python": ["python", "pip", "pandas", "numpy", "django", "flask"],
        "javascript": ["javascript", "js", "node", "react", "vue", "npm", "html", "css", "dom"],
        "java": ["java", "maven", "spring"],
        "cpp": ["c++", "cpp", "template"],
    }
    
    for lang, indicators in language_indicators.items():
        if any(ind in instruction_lower for ind in indicators):
            return lang
    
    # Default
    if any(w in instruction_lower for w in ["html", "css", "react", "dom", "browser"]):
        return "javascript"
    return "python"


def calculate_keyword_match_score(resource_keywords, question_keywords):
    """
    Calculate how many keywords match between resource and question
    
    Args:
        resource_keywords: keywords from resource (tags, topics)
        question_keywords: keywords from question
    
    Returns:
        score 0-1
    """
    if not question_keywords:
        return 0.0
    
    resource_kw_set = set(str(k).lower() for k in resource_keywords)
    question_kw_set = set(question_keywords)
    
    matches = len(resource_kw_set & question_kw_set)
    score = matches / len(question_kw_set)
    
    return min(score, 1.0)


def calculate_text_similarity(text1, text2):
    """Calculate text similarity using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    
    text1 = str(text1).lower()[:100]
    text2 = str(text2).lower()[:100]
    
    return SequenceMatcher(None, text1, text2).ratio()


def score_resource_for_question(resource, question):
    """
    Score how relevant a resource is to a question
    
    Factors:
    1. Keyword matches (most important)
    2. Language match
    3. Text similarity
    4. Resource quality (votes, answer count, etc)
    
    Returns: score 0-100
    """
    score = 0.0
    
    # Extract question keywords
    q_keywords = extract_keywords(question.get('instruction', ''))
    q_language = detect_language(question.get('instruction', ''))
    
    if not q_keywords:
        return 0.0
    
    # Scoring by resource type
    resource_type = resource.get('source', '')
    
    if resource_type == 'stackoverflow':
        # Stack Overflow scoring
        keywords_score = calculate_keyword_match_score(
            resource.get('tags', []), q_keywords
        )
        score += keywords_score * 40  # Keyword match is most important
        
        # Quality signal: votes
        votes = resource.get('votes', 0)
        votes_score = min(votes / 5000, 1.0)  # Normalize to 5000 max votes
        score += votes_score * 30
        
        # Popularity: answers
        answers = resource.get('answer_count', 0)
        answers_score = min(answers / 50, 1.0)
        score += answers_score * 10
        
        # Text match
        text_score = calculate_text_similarity(
            question.get('instruction', ''),
            resource.get('title', '')
        )
        score += text_score * 20
    
    elif resource_type == 'official_docs':
        # Official docs scoring
        keywords_score = calculate_keyword_match_score(
            resource.get('topics', []), q_keywords
        )
        score += keywords_score * 50  # Keyword match very important
        
        # Language match
        doc_language = resource.get('language', '')
        if doc_language and doc_language in q_language:
            score += 30
        
        # Text match
        text_score = calculate_text_similarity(
            question.get('instruction', ''),
            resource.get('title', '')
        )
        score += text_score * 20
    
    elif resource_type == 'github_repo':
        # GitHub repo scoring
        repo = resource.get('repo', '')
        
        # Check if repo keywords match question
        repo_keywords = repo.lower().split('/')
        repo_kw_score = calculate_keyword_match_score(
            repo_keywords, q_keywords
        )
        score += repo_kw_score * 40
        
        # Check repo title
        title_score = calculate_text_similarity(
            question.get('instruction', ''),
            resource.get('title', '')
        )
        score += title_score * 30
        
        # Language match
        if q_language in repo.lower():
            score += 30
    
    return score


def find_best_resources(question, all_resources, top_k=5):
    """
    Find top-k most relevant resources for a question
    
    Returns: list of (resource, score) tuples sorted by score
    """
    scored_resources = []
    
    for resource in all_resources:
        score = score_resource_for_question(resource, question)
        if score > 0:
            scored_resources.append((score, resource))
    
    # Sort by score descending
    scored_resources.sort(reverse=True, key=lambda x: x[0])
    
    return scored_resources[:top_k]


def load_hf_questions():
    """Load HuggingFace questions"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: datasets library not found")
        print("   Install with: pip install datasets")
        return []
    
    print("📥 Loading HuggingFace questions...")
    dataset = load_dataset("bigcode/autocodearena-v0", split="train")
    
    questions = []
    for item in dataset:
        questions.append({
            "uid": item["uid"],
            "instruction": item["instruction"],
            "category": item["category"],
        })
    
    print(f"✅ Loaded {len(questions)} HuggingFace questions")
    return questions


def load_multisource_resources():
    """Load collected multi-source resources"""
    resource_file = "autocodearena/data/new data v1.1/collected_multisource_data.json"
    
    if not os.path.exists(resource_file):
        print(f"❌ Error: {resource_file} not found")
        return []
    
    print("📥 Loading collected resources...")
    with open(resource_file, 'r', encoding='utf-8') as f:
        resources = json.load(f)
    
    print(f"✅ Loaded {len(resources)} resources")
    return resources


def create_enriched_instruction(question, resources_with_scores):
    """
    Create enriched instruction by adding best resources
    
    Args:
        question: original question dict
        resources_with_scores: list of (score, resource) tuples
    
    Returns:
        enriched instruction text
    """
    enriched = question.get('instruction', '')
    
    if not resources_with_scores:
        return enriched
    
    enriched += "\n\n" + "=" * 80
    enriched += "\n📚 **HELPFUL RESOURCES & REFERENCES**\n"
    enriched += "=" * 80 + "\n"
    
    # Group resources by type
    by_type = defaultdict(list)
    for score, resource in resources_with_scores:
        by_type[resource['source']].append((score, resource))
    
    # Stack Overflow
    if 'stackoverflow' in by_type:
        enriched += "\n### 💬 Stack Overflow Q&A\n"
        for score, resource in by_type['stackoverflow'][:2]:
            enriched += f"\n**{resource['title'][:80]}**\n"
            enriched += f"👍 {resource['votes']} votes | 💬 {resource['answer_count']} answers\n"
            enriched += f"🔗 {resource['url']}\n"
            enriched += f"_Relevance: {score:.1f}/100_\n"
    
    # Official Docs
    if 'official_docs' in by_type:
        enriched += "\n### 📖 Official Documentation\n"
        for score, resource in by_type['official_docs'][:2]:
            topics = ', '.join(resource.get('topics', [])[:3])
            enriched += f"\n**{resource['title']}**\n"
            if topics:
                enriched += f"Topics: {topics}\n"
            enriched += f"🔗 {resource['url']}\n"
            enriched += f"_Relevance: {score:.1f}/100_\n"
    
    # GitHub Examples
    if 'github_repo' in by_type:
        enriched += "\n### 🐙 Code Examples\n"
        for score, resource in by_type['github_repo'][:2]:
            enriched += f"\n**{resource['repo']}** - {resource['title']}\n"
            enriched += f"🔗 {resource['url']}\n"
            enriched += f"_Relevance: {score:.1f}/100_\n"
    
    enriched += "\n" + "=" * 80
    
    return enriched


def enrich_tasks():
    """Main enrichment function"""
    
    print("=" * 80)
    print("🚀 Enriching Tasks with Multi-Source Resources")
    print("=" * 80)
    print()
    
    # Load data
    questions = load_hf_questions()
    resources = load_multisource_resources()
    
    if not questions or not resources:
        print("❌ Failed to load questions or resources")
        return
    
    print()
    print("🔄 Mapping questions to resources...")
    print()
    
    # Enrich tasks
    enriched_tasks = []
    stats = defaultdict(lambda: {"count": 0, "avg_resources": 0})
    
    for i, question in enumerate(questions, 1):
        # Find best matching resources
        best_resources = find_best_resources(question, resources, top_k=5)
        
        # Create enriched instruction
        enriched_instruction = create_enriched_instruction(
            question, best_resources
        )
        
        # Extract keywords for metadata
        keywords = extract_keywords(question['instruction'])
        
        # Build enriched task
        enriched_task = {
            "uid": question['uid'],
            "instruction": enriched_instruction,
            "category": question['category'],
            "enrichment": {
                "original_instruction": question['instruction'],
                "mapped_resources": [
                    {
                        "source": resource['source'],
                        "title": resource['title'],
                        "url": resource['url'],
                        "score": round(score, 2),
                        "metadata": {
                            "votes": resource.get('votes'),
                            "answers": resource.get('answer_count'),
                            "topics": resource.get('topics'),
                            "tags": resource.get('tags'),
                            "repo": resource.get('repo'),
                        }
                    }
                    for score, resource in best_resources
                ],
                "num_resources": len(best_resources),
                "detected_keywords": keywords,
                "enriched_date": datetime.now().isoformat()
            }
        }
        
        enriched_tasks.append(enriched_task)
        
        # Track stats
        stats[question['category']]['count'] += 1
        stats[question['category']]['avg_resources'] += len(best_resources)
        
        if i % 20 == 0:
            print(f"  ✓ Enriched {i}/{len(questions)} tasks")
    
    # Calculate averages
    for cat in stats:
        count = stats[cat]['count']
        stats[cat]['avg_resources'] = round(stats[cat]['avg_resources'] / count, 2)
    
    print()
    print("=" * 80)
    print("📊 ENRICHMENT SUMMARY")
    print("=" * 80)
    print(f"HuggingFace questions: {len(questions)}")
    print(f"Available resources: {len(resources)}")
    print(f"Enriched tasks: {len(enriched_tasks)}")
    print()
    print("By Category:")
    for cat in sorted(stats.keys()):
        s = stats[cat]
        print(f"  {cat:25s}: {s['count']:3d} tasks, "
              f"avg {s['avg_resources']:.1f} resources each")
    print()
    
    # Save enriched tasks
    output_file = "autocodearena/data/new data v1.1/tasks_multisource.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enriched tasks saved to: {output_file}")
    print()
    
    # Show sample
    print("=" * 80)
    print("📋 SAMPLE ENRICHED TASK")
    print("=" * 80)
    print()
    
    if enriched_tasks:
        sample = enriched_tasks[0]
        print(f"Task UID: {sample['uid']}")
        print(f"Category: {sample['category']}")
        print(f"Resources found: {sample['enrichment']['num_resources']}")
        print(f"Keywords: {', '.join(sample['enrichment']['detected_keywords'][:5])}")
        print()
        print("Resources mapped:")
        for res in sample['enrichment']['mapped_resources']:
            print(f"  • {res['source']:15s} - {res['title'][:50]}... (score: {res['score']})")
        print()
        print("Enriched instruction (first 600 chars):")
        print("-" * 80)
        print(sample['instruction'][:600])
        print("...")
    
    print()
    print("=" * 80)
    print("✨ Enrichment complete!")
    print("=" * 80)
    
    return enriched_tasks


if __name__ == "__main__":
    try:
        enrich_tasks()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
