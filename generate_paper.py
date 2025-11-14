"""
Generate a flawed scientific paper for the review task
"""
import random
import json


def generate_flawed_paper(seed: int = 42) -> dict:
    """
    Generate a scientific paper with intentional flaws for review
    
    Returns a dictionary with:
    - title: Paper title
    - abstract: Paper abstract
    - introduction: Introduction section
    - methodology: Methods section
    - results: Results section
    - conclusion: Conclusion section
    - references: List of references
    - metadata: Paper metadata
    """
    random.seed(seed)
    
    # Choose a random topic
    topics = [
        {
            "title": "Novel Approach to Sentiment Analysis Using Quantum Computing",
            "field": "NLP",
            "claim": "98% accuracy on sentiment classification",
            "model": "Quantum-BERT"
        },
        {
            "title": "Breakthrough in Image Recognition: Zero-Shot Learning with 99.9% Accuracy",
            "field": "Computer Vision",
            "claim": "99.9% accuracy on ImageNet",
            "model": "MegaVision-X"
        },
        {
            "title": "Revolutionary Machine Learning Algorithm Solves P vs NP",
            "field": "Theory",
            "claim": "proves P=NP using neural networks",
            "model": "DeepProof"
        }
    ]
    
    topic = random.choice(topics)
    
    # Introduce random flaws
    flaws = []
    
    # Flaw 1: Dataset issues (80% chance)
    dataset_flaw = None
    if random.random() < 0.8:
        dataset_flaw = random.choice([
            "Dataset size not mentioned",
            "No train/test split described",
            "Data leakage between train and test",
            "Biased sampling method",
            "Dataset not publicly available or referenced"
        ])
        flaws.append(("dataset", dataset_flaw))
    
    # Flaw 2: Methodology issues (90% chance)
    method_flaw = None
    if random.random() < 0.9:
        method_flaw = random.choice([
            "Hyperparameters not specified",
            "No baseline comparison",
            "Cherry-picked metrics",
            "Unclear model architecture",
            "No ablation study"
        ])
        flaws.append(("methodology", method_flaw))
    
    # Flaw 3: Statistical issues (70% chance)
    stats_flaw = None
    if random.random() < 0.7:
        stats_flaw = random.choice([
            "No confidence intervals",
            "Single run, no error bars",
            "P-values not reported",
            "No significance testing",
            "Results not reproducible"
        ])
        flaws.append(("statistics", stats_flaw))
    
    # Flaw 4: Claims issues (85% chance)
    claim_flaw = None
    if random.random() < 0.85:
        claim_flaw = random.choice([
            "Overclaimed results",
            "Unrealistic accuracy claims",
            "No comparison to state-of-the-art",
            "Contradictory statements",
            "Missing important limitations"
        ])
        flaws.append(("claims", claim_flaw))
    
    # Flaw 5: Reference issues (60% chance)
    ref_flaw = None
    if random.random() < 0.6:
        ref_flaw = random.choice([
            "Key papers not cited",
            "Self-citations only",
            "Outdated references (pre-2015)",
            "Missing related work",
            "References not properly formatted"
        ])
        flaws.append(("references", ref_flaw))
    
    # Generate paper content with flaws
    abstract = f"""
We present {topic['model']}, a novel approach to {topic['field']} that achieves {topic['claim']}. 
Our method outperforms all existing approaches by a significant margin. 
{dataset_flaw if dataset_flaw and 'Dataset' in dataset_flaw else 'We trained on a large dataset.'}
The results demonstrate the superiority of our approach and open new avenues for research.
"""
    
    introduction = f"""
1. Introduction

{topic['field']} has been an active area of research for decades. Recent advances in deep learning
have shown promising results. However, existing methods still face limitations.

{claim_flaw if claim_flaw and 'state-of-the-art' in claim_flaw else ''}

In this paper, we propose {topic['model']}, which addresses these limitations and achieves
unprecedented performance. Our contributions are:
1. A novel architecture
2. State-of-the-art results
3. Extensive experiments

{ref_flaw if ref_flaw and 'Key papers' in ref_flaw else 'Recent work by Smith et al. (2023) explored similar ideas.'}
"""
    
    methodology = f"""
2. Methodology

Our approach is based on {random.choice(['transformer', 'CNN', 'RNN', 'hybrid'])} architecture.
{method_flaw if method_flaw else 'We carefully tuned all hyperparameters.'}

2.1 Model Architecture
The model consists of multiple layers. {method_flaw if method_flaw and 'architecture' in method_flaw else 'Details are in the appendix.'}

2.2 Training Procedure
We trained the model using {random.choice(['Adam', 'SGD', 'AdamW'])} optimizer.
{dataset_flaw if dataset_flaw and 'split' in dataset_flaw else 'Standard 80/20 train/test split was used.'}

2.3 Evaluation
{stats_flaw if stats_flaw else 'We evaluated on standard benchmarks with proper statistical testing.'}
"""
    
    results = f"""
3. Results

Our method achieves {topic['claim']} on the test set.
{stats_flaw if stats_flaw and 'Single run' in stats_flaw else 'Results are averaged over 5 runs.'}

Table 1: Performance Comparison
Method          Accuracy
Baseline        {random.uniform(0.75, 0.85):.2f}
{topic['model']} {random.uniform(0.95, 0.99):.2f}

{method_flaw if method_flaw and 'No baseline' in method_flaw else 'Compared against strong baselines.'}
{claim_flaw if claim_flaw and 'Overclaimed' in claim_flaw else 'Results demonstrate clear improvements.'}
"""
    
    conclusion = f"""
4. Conclusion

We presented {topic['model']}, achieving {topic['claim']}.
{claim_flaw if claim_flaw and 'limitations' in claim_flaw else 'Some limitations remain for future work.'}

Our work opens exciting new directions for {topic['field']} research.
"""
    
    references = """
5. References

[1] Smith, J. et al. (2023). "Some Related Work"
[2] Jones, A. (2022). "Another Paper"
""" + (f"\n{ref_flaw}" if ref_flaw and 'formatted' in ref_flaw else "")
    
    paper = {
        "title": topic['title'],
        "abstract": abstract.strip(),
        "introduction": introduction.strip(),
        "methodology": methodology.strip(),
        "results": results.strip(),
        "conclusion": conclusion.strip(),
        "references": references.strip(),
        "metadata": {
            "field": topic['field'],
            "claim": topic['claim'],
            "model": topic['model']
        },
        "intentional_flaws": flaws,
        "flaw_categories": {
            "dataset": dataset_flaw,
            "methodology": method_flaw,
            "statistics": stats_flaw,
            "claims": claim_flaw,
            "references": ref_flaw
        }
    }
    
    return paper


def save_paper_to_file(paper: dict, filename: str = "paper_to_review.json"):
    """Save paper to JSON file"""
    # Don't include intentional_flaws and flaw_categories in saved file
    paper_for_save = {
        "title": paper["title"],
        "abstract": paper["abstract"],
        "introduction": paper["introduction"],
        "methodology": paper["methodology"],
        "results": paper["results"],
        "conclusion": paper["conclusion"],
        "references": paper["references"],
        "metadata": paper["metadata"]
    }
    
    with open(filename, 'w') as f:
        json.dump(paper_for_save, f, indent=2)
    
    print(f"Generated paper: {filename}")
    print(f"Title: {paper['title']}")
    print(f"\nIntentional flaws introduced:")
    for category, flaw in paper['flaw_categories'].items():
        if flaw:
            print(f"  - {category}: {flaw}")
    print(f"\nTotal flaws: {len([f for f in paper['flaw_categories'].values() if f])}")
    
    return paper_for_save


if __name__ == "__main__":
    paper = generate_flawed_paper(seed=42)
    save_paper_to_file(paper)
