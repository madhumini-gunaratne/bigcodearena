import pandas as pd
import argparse
import os
import torch
import json
from glob import glob
from tqdm import tqdm

from utils.math_utils import one_hot_encode, to_winrate_probabilities, bootstrap_pairwise_model
from utils.completion import make_config, load_questions_from_hf

def load_category_info(dataset_repo_id):
    """Load category information from HuggingFace dataset"""
    print(f"Loading category information from HuggingFace dataset: {dataset_repo_id}...")
    category_info = {}
    
    try:
        questions = load_questions_from_hf(repo_id=dataset_repo_id)
        for question in questions:
            uid = question.get('uid')
            category = question.get('category')
            
            if uid:
                category_info[uid] = category
        
        print(f"Loaded category information for {len(category_info)} questions")
        
        # Show available categories
        unique_categories = set(category_info.values())
        print(f"Available categories: {sorted(unique_categories)}")
        
    except Exception as e:
        print(f"Error loading category information from HuggingFace: {e}")
    
    return category_info

def load_judgments(judge_names, benchmark, weight=3, no_execution_results=False):
    dfs = []
    for judge_name in judge_names:
        print(f"Loading {judge_name} judgments...")
        if no_execution_results:
            judgment_dir = "model_judgment_no_results"
        else:
            judgment_dir = "model_judgment"
            
        dfs.extend([
            pd.read_json(f, lines=True) for f in tqdm(glob(os.path.join(
                "data",
                benchmark, 
                judgment_dir, 
                judge_name, 
                "*.jsonl"
            )))
        ])
    data = pd.concat(dfs).reset_index(drop=True)
    
    # if data.model.isin(judge_names).any():
    #     print(f"WARNING: {judge_names} is already in the data. Removing it.")
    #     data = data[~data.model.isin(judge_names)].reset_index(drop=True)

    null_indices = data.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
    _data = data[~null_indices].reset_index(drop=True)
    
    print(f"Number of null judgments found: {len(data) - len(_data)}")
    
    # map label to score
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    _data['scores'] = _data.games.map(
        lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    )
    
    battles = _data[['uid', 'model', 'environment', 'scores']].explode('scores').reset_index(drop=True)
    
    return battles


def get_model_style_metadata(benchmark):
    model_metadata = {}
    for file in glob(os.path.join("data", benchmark, "model_answer", "*.jsonl")):
        df = pd.read_json(file, lines=True)
        model_metadata[df.iloc[0]['model']] = df.set_index('uid')['metadata'].to_dict()
        
    return model_metadata


def format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline=None):
    leaderboard = pd.merge(
        mean_scores, 
        lower_scores, 
        on="model"
    ).merge(
        upper_scores, 
        on="model"
    )
    
    leaderboard["Scores (%)"] = leaderboard["scores"].map(lambda x: round(x * 100, 1))
    
    # Create CI as tuples: (lower_bound, upper_bound) in percentage points
    leaderboard["CI (%)"] = leaderboard.apply(
        lambda row: (-round((row['scores'] - row['lower']) * 100, 1), round((row['upper'] - row['scores']) * 100, 1)), 
        axis=1
    )
    
    _leaderboard = leaderboard.rename(
        columns={"model": "Model"}
    ).drop(
        columns=["lower", "upper", "scores"]
    )
    
    if baseline and baseline not in _leaderboard["Model"].values:
        baseline_df = pd.DataFrame({
            "Model": [baseline], 
            "Scores (%)": [50.0], 
            "CI (%)": [(0.0, 0.0)]
        })
        _leaderboard = pd.concat([_leaderboard, baseline_df], ignore_index=True)
    
    return _leaderboard.sort_values(by="Scores (%)", ascending=False).reset_index(drop=True)


def print_leaderboard(battles, topic, print_output=True):
    baseline = BASELINE_MODEL

    _battles = battles.drop(columns=['topic'])[['model', 'scores']]
    
    # remove model path
    _battles['model'] = _battles['model'].map(lambda x: x.split('/')[-1])
    
    bootstraps = pd.concat([
        _battles.groupby("model").sample(frac=1.0, replace=True).groupby("model").mean()
        for _ in tqdm(range(100))
    ])
    
    bootstraps["scores"] = bootstraps["scores"].astype(float)
    
    mean_scores = bootstraps.groupby("model").mean().reset_index()
    lower_scores = bootstraps.groupby("model").quantile(0.05).reset_index().rename(columns={"scores": "lower"})
    upper_scores = bootstraps.groupby("model").quantile(0.95).reset_index().rename(columns={"scores": "upper"})
    
    _leaderboard = format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline)
    
    if print_output:
        print(f"##### Topic: {topic} #####")
        print(_leaderboard.to_string())
    
    return _leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", type=str, default="autocodearena")
    parser.add_argument("--setting-file", "-s", type=str, default="config/autocodearena.yaml")
    parser.add_argument("--dataset", type=str, default="bigcode/autocodearena-v0", help="HuggingFace dataset repository ID")
    parser.add_argument("--judge-names", "-j", nargs="+", default=["claude37_sonnet"])
    parser.add_argument("--control-features", "-f", nargs="+", default=[])
    parser.add_argument("--topic", "-t", nargs="+", default=[], help="Filter by specific topics (category names)")
    parser.add_argument("--no-execution-results", action="store_true", help="Load judgments from model_judgment_no_results directory (without execution results)")
    parser.add_argument("--by", choices=["topic", "env"], default="topic", help="Display results by topic or environment (default: topic)")
    parser.add_argument("--latex", choices=["rank", "score"], default=None, help="Output LaTeX table: 'rank' for rank table, 'score' for score table")
    parser.add_argument("--json", type=str, help="Output results to JSON file (specify filename)")
    args = parser.parse_args()
    
    configs = make_config(args.setting_file)
    BASELINE_MODEL = configs["baseline_model"]

    # Load category information from HuggingFace dataset
    category_info = load_category_info(args.dataset)
    
    all_battles = load_judgments(args.judge_names, args.benchmark, no_execution_results=args.no_execution_results)
    print("Number of battles loaded: ", len(all_battles))

    # Add topic information to battles
    if category_info:
        all_battles['topic'] = all_battles['uid'].map(category_info)
        # Remove battles without topic information
        all_battles = all_battles.dropna(subset=['topic']).reset_index(drop=True)
        print(f"Number of battles with topic information: {len(all_battles)}")
    else:
        print("No category information available. Cannot split by topic.")
        exit(1)

    # Normalize whitespace in categorical fields
    if "environment" in all_battles.columns:
        all_battles["environment"] = all_battles["environment"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    if "topic" in all_battles.columns:
        all_battles["topic"] = all_battles["topic"].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Helper utilities for LaTeX output
    def _latex_escape(text: str) -> str:
        return text.replace("_", "\\_") if isinstance(text, str) else text

    def _compute_mean_scores_by_model(df: pd.DataFrame) -> pd.DataFrame:
        _battles = df[["model", "scores"]].copy()
        _battles["model"] = _battles["model"].map(lambda x: x.split("/")[-1])
        _battles["scores"] = pd.to_numeric(_battles["scores"], errors="coerce")
        mean_scores = _battles.groupby("model").mean(numeric_only=True).reset_index()
        mean_scores = mean_scores.rename(columns={"model": "Model", "scores": "score"})
        return mean_scores

    def _generate_latex_table(all_df: pd.DataFrame, by_field: str, groups: list[str], mode: str) -> str:
        # Aggregate per group
        per_group_frames = []
        filtered = all_df[all_df[by_field].isin(groups)].reset_index(drop=True)
        for g in groups:
            g_df = filtered[filtered[by_field] == g]
            if len(g_df) == 0:
                continue
            ms = _compute_mean_scores_by_model(g_df)
            ms = ms.rename(columns={"score": g})
            per_group_frames.append(ms)

        if not per_group_frames:
            return "% No data available to generate LaTeX table."

        merged = per_group_frames[0]
        for frame in per_group_frames[1:]:
            merged = pd.merge(merged, frame, on="Model", how="outer")

        # Overall on filtered subset
        overall_scores = _compute_mean_scores_by_model(filtered).rename(columns={"score": "Overall"})
        merged = pd.merge(merged, overall_scores, on="Model", how="outer")

        # Prepare values: only include requested groups that are available
        available_groups = [g for g in groups if g in merged.columns]
        include_overall = "Overall" in merged.columns
        header_names = [*available_groups, *( ["Overall"] if include_overall else [] )]
        if not header_names:
            return "% No data columns available to generate LaTeX table."
        value_df = merged.set_index("Model")[header_names]
        if mode == "rank":
            ranks = value_df.rank(axis=0, ascending=False, method="min")
            # Convert to int where not NaN
            ranks = ranks.applymap(lambda x: int(x) if pd.notna(x) else None)
            table_values = ranks
            # Order models by Overall ascending rank
            if include_overall and "Overall" in ranks.columns:
                sort_index = ranks["Overall"].sort_values(na_position="last").index
            else:
                sort_index = ranks.mean(axis=1, numeric_only=True).sort_values(na_position="last").index
            table_values = table_values.loc[sort_index]
        else:  # score
            # Convert to percentage with 1 decimal place
            scores_pct = value_df.applymap(lambda x: round(x * 100, 1) if pd.notna(x) else None)
            table_values = scores_pct
            # Order models by Overall descending score
            if include_overall and "Overall" in scores_pct.columns:
                sort_index = scores_pct["Overall"].sort_values(ascending=False, na_position="last").index
            else:
                sort_index = scores_pct.mean(axis=1, numeric_only=True).sort_values(ascending=False, na_position="last").index
            table_values = table_values.loc[sort_index]

        # Build LaTeX
        title = "Model Ranks Across Different " + ("Programming Topics" if by_field == "topic" else "Environments") if mode == "rank" else "Model Scores (%) Across Different " + ("Programming Topics" if by_field == "topic" else "Environments")
        label = "tab:model_ranks" if mode == "rank" else "tab:model_scores"

        # Column spec: l | c... | c
        num_group_cols = len(available_groups)
        col_spec = "l|" + ("c" * num_group_cols) + ("|c" if include_overall else "")
        header_cols = " & ".join([f"\\textbf{{{_latex_escape(h)}}}" for h in header_names])

        lines = []
        lines.append("\\begin{table}[!t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{title}}}")
        lines.append("\\resizebox{\\linewidth}{!}{")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(f"\\textbf{{Model}} & {header_cols} \\\\")
        lines.append("\\midrule")

        for model, row in table_values.iterrows():
            model_tex = _latex_escape(model)
            cells = []
            for g in header_names:
                val = row.get(g)
                if pd.isna(val):
                    cells.append("")
                else:
                    cells.append(str(int(val)) if mode == "rank" else (f"{val:.1f}" if isinstance(val, float) else str(val)))
            lines.append(f"{model_tex} & " + " & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    if args.latex is not None:
        # Prepare group list
        if args.by == "topic":
            groups = args.topic if args.topic else sorted(all_battles.topic.unique())
            latex = _generate_latex_table(all_battles, by_field="topic", groups=groups, mode=args.latex)
        else:
            environments = all_battles.environment.unique()
            environments = [env for env in environments if env]
            groups = sorted(environments)
            latex = _generate_latex_table(all_battles, by_field="environment", groups=groups, mode=args.latex)

        print(latex)
        exit(0)

    if args.json:
        # Prepare data for JSON output
        data_for_json = {}
        
        if args.by == "topic":
            topics = args.topic if args.topic else sorted(all_battles.topic.unique())
            for topic in topics:
                battles = all_battles[all_battles.topic == topic].reset_index(drop=True)
                leaderboard_data = print_leaderboard(battles, topic, print_output=False)
                data_for_json[topic] = leaderboard_data.to_dict(orient='records')
            
            # Add overall leaderboard
            overall_battles = all_battles.copy()
            overall_leaderboard = print_leaderboard(overall_battles, "All Topics", print_output=False)
            data_for_json["Overall"] = overall_leaderboard.to_dict(orient='records')
        else:
            # Display by environment
            environments = all_battles.environment.unique()
            environments = [env for env in environments if env]
            
            for env in sorted(environments):
                battles = all_battles[all_battles.environment == env].reset_index(drop=True)
                leaderboard_data = print_leaderboard(battles, f"Environment: {env}", print_output=False)
                data_for_json[env] = leaderboard_data.to_dict(orient='records')
            
            # Add overall leaderboard
            overall_battles = all_battles.copy()
            overall_leaderboard = print_leaderboard(overall_battles, "All Environments", print_output=False)
            data_for_json["Overall"] = overall_leaderboard.to_dict(orient='records')
        
        # Output JSON
        with open(args.json, 'w') as f:
            json.dump(data_for_json, f, indent=2)
        print(f"Results saved to {args.json}")
        exit(0)

    if args.by == "topic":
        # Display by topic (original behavior)
        topics = args.topic if args.topic else all_battles.topic.unique()
        
        # Print leaderboard presentation configs
        print("=" * 50)
        print("Leaderboard Presentation Configs:\n")

        print(f"Benchmark: {args.benchmark}")
        print(f"Control features: {args.control_features}")
        print(f"Baseline model: {BASELINE_MODEL}")
        print(f"Judge names: {args.judge_names}")
        print(f"Judgment type: {'No execution results' if args.no_execution_results else 'With execution results'}")
        print(f"Display mode: {args.by}")
        print(f"Number of topics in battle: {len(all_battles.topic.unique())}")
        print(f"    {sorted(all_battles.topic.unique())}")
        print(f"Number of topics to show: {len(topics)}")
        print(f"    {topics}")
        print(f"Number of models: {len(all_battles.model.unique())}")
        print(f"Models: {all_battles.model.unique()}")
        print(f"Number of battles: {len(all_battles)}")
        print("=" * 50)
        
        for topic in topics:
            assert topic in all_battles.topic.unique(), f"Invalid topic: {topic}"

            battles = all_battles[all_battles.topic == topic].reset_index(drop=True)
            leaderboard_data = print_leaderboard(battles, topic)

        print("=" * 50)
        print("Final Leaderboard:")
        leaderboard_data = print_leaderboard(all_battles, "All Topics")
        
    else:
        # Display by environment
        environments = all_battles.environment.unique()
        environments = [env for env in environments if env]
        
        # Print leaderboard presentation configs
        print("=" * 50)
        print("Leaderboard Presentation Configs:\n")

        print(f"Benchmark: {args.benchmark}")
        print(f"Control features: {args.control_features}")
        print(f"Baseline model: {BASELINE_MODEL}")
        print(f"Judge names: {args.judge_names}")
        print(f"Judgment type: {'No execution results' if args.no_execution_results else 'With execution results'}")
        print(f"Display mode: {args.by}")
        print(f"Number of environments in battle: {len(environments)}")
        print(f"    {sorted(environments)}")
        print(f"Number of models: {len(all_battles.model.unique())}")
        print(f"Models: {all_battles.model.unique()}")
        print(f"Number of battles: {len(all_battles)}")
        print("=" * 50)
        
        for env in sorted(environments):
            battles = all_battles[all_battles.environment == env].reset_index(drop=True)
            leaderboard_data = print_leaderboard(battles, f"Environment: {env}")

        print("=" * 50)
        print("Final Leaderboard:")
        leaderboard_data = print_leaderboard(all_battles, "All Environments")
        