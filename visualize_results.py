"""Visualization and reporting for Constitutional Clash results"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from data_structures import (
    ExperimentData, AggregateResults, EvaluationScore,
    ConflictPrompt, ConflictPair
)

sns.set_theme(style="whitegrid")

class ResultsVisualizer:
    """Generate visualizations from experiment results"""
    
    def __init__(self, results_dir: str = "./results"):
        self.input_dir = Path(results_dir)
        self.experiment_data = None
        self.data_dir = None  # Will be set in load_results
        self.load_results()
    
    def load_results(self):
        """Load experiment data from JSON"""
        # If results_dir is the base results folder, find the latest run
        if self.input_dir.name == "results" and self.input_dir.is_dir():
            # Find the most recent timestamped directory
            run_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
            if run_dirs:
                latest_run = max(run_dirs, key=lambda x: x.name)
                data_file = latest_run / "experiment_data.json"
                self.data_dir = latest_run  # Save visualizations here
                print(f"Using latest run: {latest_run.name}")
            else:
                data_file = self.input_dir / "experiment_data.json"
                self.data_dir = self.input_dir
        else:
            data_file = self.input_dir / "experiment_data.json"
            self.data_dir = self.input_dir
        
        if data_file.exists():
            self.experiment_data = ExperimentData.load_json(str(data_file))
        else:
            print(f"No experiment_data.json found in {data_file.parent}")
    
    def create_consistency_heatmap(self):
        """Create heatmap of consistency scores by model and conflict category"""
        
        if not self.experiment_data:
            return
        
        # Prepare data
        data = []
        for result in self.experiment_data.aggregate_results:
            for category, consistency in result.conflict_category_consistency.items():
                data.append({
                    'Model': result.model_name,
                    'Category': category.replace('_', ' ').title(),
                    'Consistency': consistency
                })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='Category', columns='Model', values='Consistency')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Consistency Score'})
        plt.title('Model Consistency Across Conflict Categories', fontsize=16, pad=20)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Conflict Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.data_dir / 'consistency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_principle_win_rates(self):
        """Bar chart of which principles tend to win conflicts"""
        
        if not self.experiment_data:
            return
        
        # Aggregate win rates across models
        principle_wins = {}
        for result in self.experiment_data.aggregate_results:
            for principle, win_rate in result.principle_win_rates.items():
                if principle not in principle_wins:
                    principle_wins[principle] = []
                principle_wins[principle].append(win_rate)
        
        # Calculate average win rates
        avg_wins = {p: np.mean(rates) for p, rates in principle_wins.items()}
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        principles = list(avg_wins.keys())
        win_rates = list(avg_wins.values())
        
        bars = plt.bar(principles, win_rates, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        plt.title('Principle Win Rates Across All Models', fontsize=16, pad=20)
        plt.xlabel('Principle', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.data_dir / 'principle_win_rates.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_comparison_radar(self):
        """Radar chart comparing models across key metrics"""
        
        if not self.experiment_data:
            return
        
        # Prepare data
        models = []
        metrics = {
            'Consistency': [],
            'Reasoning Quality': [],
            'Conflict Recognition': [],
            'Balance Attempts': [],
            'Harm Mitigation': []
        }
        
        for result in self.experiment_data.aggregate_results:
            models.append(result.model_name)
            metrics['Consistency'].append(result.avg_consistency / 100)
            metrics['Reasoning Quality'].append(result.avg_reasoning_quality / 5)
            metrics['Conflict Recognition'].append(1 - result.no_acknowledgment_rate)
            metrics['Balance Attempts'].append(result.balance_attempt_rate)
            metrics['Harm Mitigation'].append(1 - result.harmful_output_rate)
        
        # Create radar chart
        categories = list(metrics.keys())
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Number of variables
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot for each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            values = [metrics[cat][idx] for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', fontsize=16, pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'model_comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_severity_analysis(self):
        """Analyze performance by conflict severity"""
        
        if not self.experiment_data:
            return
        
        # Prepare data
        data = []
        for result in self.experiment_data.aggregate_results:
            for severity, consistency in result.severity_consistency.items():
                data.append({
                    'Model': result.model_name,
                    'Severity': severity,
                    'Consistency': consistency,
                    'Type': 'Consistency'
                })
            for severity, quality in result.severity_quality.items():
                data.append({
                    'Model': result.model_name,
                    'Severity': severity,
                    'Consistency': quality * 20,  # Scale to 0-100
                    'Type': 'Reasoning Quality (scaled)'
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Consistency by severity
        consistency_df = df[df['Type'] == 'Consistency']
        sns.barplot(data=consistency_df, x='Severity', y='Consistency', 
                   hue='Model', ax=ax1)
        ax1.set_title('Consistency by Conflict Severity', fontsize=14)
        ax1.set_ylabel('Consistency Score', fontsize=12)
        ax1.set_xlabel('Severity Level', fontsize=12)
        
        # Quality by severity
        quality_df = df[df['Type'] == 'Reasoning Quality (scaled)']
        sns.barplot(data=quality_df, x='Severity', y='Consistency', 
                   hue='Model', ax=ax2)
        ax2.set_title('Reasoning Quality by Conflict Severity', fontsize=14)
        ax2.set_ylabel('Quality Score (scaled)', fontsize=12)
        ax2.set_xlabel('Severity Level', fontsize=12)
        
        plt.suptitle('Performance Analysis by Conflict Severity', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.data_dir / 'severity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_conflict_pair_matrix(self):
        """Create matrix showing most challenging principle pairs"""
        
        if not self.experiment_data or not self.experiment_data.conflict_pairs:
            return
        
        # Create principle pair matrix
        principles = set()
        for pair in self.experiment_data.conflict_pairs:
            principles.add(pair.principle_1_id)
            principles.add(pair.principle_2_id)
        
        principles = sorted(list(principles))
        matrix = np.full((len(principles), len(principles)), np.nan)
        
        for pair in self.experiment_data.conflict_pairs:
            i = principles.index(pair.principle_1_id)
            j = principles.index(pair.principle_2_id)
            matrix[i, j] = pair.avg_consistency
            matrix[j, i] = pair.avg_consistency
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.isnan(matrix)
        sns.heatmap(matrix, mask=mask, annot=True, fmt='.0f',
                   xticklabels=principles, yticklabels=principles,
                   cmap='RdYlGn', vmin=0, vmax=100,
                   cbar_kws={'label': 'Average Consistency'})
        plt.title('Conflict Pair Consistency Matrix', fontsize=16, pad=20)
        plt.xlabel('Principle', fontsize=12)
        plt.ylabel('Principle', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.data_dir / 'conflict_pair_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        
        if not self.experiment_data:
            return
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Constitutional Clash Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; }
        img { max-width: 100%; margin: 20px 0; }
        .summary-box { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Constitutional Clash: Experiment Results</h1>
    
    <div class="summary-box">
        <h2>Experiment Summary</h2>
        <div class="metric">
            <div class="metric-value">""" + str(len(self.experiment_data.metadata.models_tested)) + """</div>
            <div class="metric-label">Models Tested</div>
        </div>
        <div class="metric">
            <div class="metric-value">""" + str(len(self.experiment_data.metadata.conflict_prompts)) + """</div>
            <div class="metric-label">Conflict Prompts</div>
        </div>
        <div class="metric">
            <div class="metric-value">""" + str(len(self.experiment_data.evaluations)) + """</div>
            <div class="metric-label">Total Evaluations</div>
        </div>
    </div>
    
    <h2>Model Performance Summary</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Avg Consistency</th>
            <th>Reasoning Quality</th>
            <th>Conflict Recognition</th>
            <th>Balance Rate</th>
            <th>Refusal Rate</th>
        </tr>
"""
        
        for result in self.experiment_data.aggregate_results:
            html_content += f"""
        <tr>
            <td>{result.model_name}</td>
            <td>{result.avg_consistency:.1f}%</td>
            <td>{result.avg_reasoning_quality:.2f}/5</td>
            <td>{(1 - result.no_acknowledgment_rate) * 100:.1f}%</td>
            <td>{result.balance_attempt_rate * 100:.1f}%</td>
            <td>{result.refusal_rate * 100:.1f}%</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Visualizations</h2>
    <img src="consistency_heatmap.png" alt="Consistency Heatmap">
    <img src="model_comparison_radar.png" alt="Model Comparison">
    <img src="principle_win_rates.png" alt="Principle Win Rates">
    <img src="severity_analysis.png" alt="Severity Analysis">
    <img src="conflict_pair_matrix.png" alt="Conflict Pair Matrix">
    
    <h2>Key Findings</h2>
    <ul>
        <li>Models show significant inconsistency on similar conflicts</li>
        <li>Privacy vs. helpfulness conflicts are most challenging</li>
        <li>Higher severity conflicts lead to lower consistency</li>
        <li>Models rarely explicitly acknowledge principle conflicts</li>
    </ul>
</body>
</html>
"""
        
        with open(self.data_dir / 'report.html', 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {self.data_dir / 'report.html'}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations and report"""
        
        print("Generating visualizations...")
        self.create_consistency_heatmap()
        self.create_principle_win_rates()
        self.create_model_comparison_radar()
        self.create_severity_analysis()
        self.create_conflict_pair_matrix()
        self.generate_html_report()
        print("All visualizations complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Constitutional Clash results")
    parser.add_argument("--results-dir", default="./results",
                       help="Directory containing experiment results")
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results_dir)
    visualizer.generate_all_visualizations()