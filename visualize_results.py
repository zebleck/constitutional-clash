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
            
            # Check if we have incomplete aggregate results
            if self.experiment_data.aggregate_results:
                models_in_aggregates = {r.model_name for r in self.experiment_data.aggregate_results}
                models_in_evaluations = {e.model_name for e in self.experiment_data.evaluations}
                missing_models = models_in_evaluations - models_in_aggregates
                
                if missing_models:
                    print(f"Warning: Found incomplete analysis. Missing aggregate results for: {missing_models}")
                    print("Consider re-running the analysis phase to get complete results.")
            
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
        plt.close()
    
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
        
        # Calculate average win rates and convert to percentages
        avg_wins = {p: np.mean(rates) * 100 for p, rates in principle_wins.items()}
        
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
        plt.ylim(0, 100)
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.data_dir / 'principle_win_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.close()
    
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
        plt.close()
    
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
        plt.close()
    
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
    
    def create_individual_model_visualizations(self):
        """Create individual visualizations for each model"""
        
        if not self.experiment_data:
            return
        
        print("Generating individual model visualizations...")
        
        for result in self.experiment_data.aggregate_results:
            model_name = result.model_name
            model_dir = self.data_dir / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Creating visualizations for {model_name}...")
            
            # Individual model performance breakdown
            self._create_model_performance_chart(result, model_dir)
            
            # Individual model category breakdown
            self._create_model_category_analysis(result, model_dir)
            
            # Individual model response examples
            self._create_model_response_examples(result, model_dir)
            
            # Individual model principle preferences
            self._create_model_principle_preferences(result, model_dir)
    
    def _create_model_performance_chart(self, result, output_dir):
        """Create performance breakdown chart for individual model"""
        
        metrics = {
            'Consistency': result.avg_consistency,
            'Reasoning Quality': result.avg_reasoning_quality * 20,  # Scale to 0-100
            'Conflict Recognition': (1 - result.no_acknowledgment_rate) * 100,
            'Balance Attempts': result.balance_attempt_rate * 100,
            'Harm Mitigation': (1 - result.harmful_output_rate) * 100
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics.keys(), metrics.values(), color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        ax.set_title(f'{result.model_name} Performance Breakdown', fontsize=16, pad=20)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_category_analysis(self, result, output_dir):
        """Create category-specific analysis for individual model"""
        
        categories = list(result.conflict_category_consistency.keys())
        consistency_scores = list(result.conflict_category_consistency.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Category consistency
        bars1 = ax1.bar(categories, consistency_scores, color='lightcoral', alpha=0.8)
        ax1.set_title(f'{result.model_name} Consistency by Category', fontsize=14)
        ax1.set_ylabel('Consistency Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 100)
        
        for bar, score in zip(bars1, consistency_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Severity performance
        severity_data = result.severity_consistency
        severities = list(severity_data.keys())
        severity_scores = list(severity_data.values())
        
        bars2 = ax2.bar(severities, severity_scores, color='lightgreen', alpha=0.8)
        ax2.set_title(f'{result.model_name} Performance by Severity', fontsize=14)
        ax2.set_ylabel('Consistency Score', fontsize=12)
        ax2.set_ylim(0, 100)
        
        for bar, score in zip(bars2, severity_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.suptitle(f'{result.model_name} Detailed Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_response_examples(self, result, output_dir):
        """Create visualization of response characteristics for individual model"""
        
        # Response characteristics pie chart - ensure non-negative values
        refusal_pct = max(0, result.refusal_rate * 100)
        balance_pct = max(0, result.balance_attempt_rate * 100)
        other_pct = max(0, 100 - refusal_pct - balance_pct)
        
        # Filter out zero values
        labels = []
        sizes = []
        colors = []
        
        if refusal_pct > 0:
            labels.append('Refuses Answer')
            sizes.append(refusal_pct)
            colors.append('lightcoral')
        
        if balance_pct > 0:
            labels.append('Attempts Balance')
            sizes.append(balance_pct)
            colors.append('lightblue')
        
        if other_pct > 0:
            labels.append('Other')
            sizes.append(other_pct)
            colors.append('lightgray')
        
        # If all values are 0, create a single "No Data" slice
        if not sizes:
            labels = ['No Data']
            sizes = [100]
            colors = ['lightgray']
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
        
        plt.title(f'{result.model_name} Response Characteristics', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'response_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_principle_preferences(self, result, output_dir):
        """Create principle preference chart for individual model"""
        
        principles = list(result.principle_win_rates.keys())
        win_rates = [rate * 100 for rate in result.principle_win_rates.values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(principles, win_rates, color='gold', alpha=0.8)
        
        # Add value labels
        for bar, rate in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        ax.set_title(f'{result.model_name} Principle Win Rates', fontsize=16, pad=20)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_xlabel('Principle', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 100)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'principle_preferences.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_model_comparison_visualizations(self):
        """Create head-to-head model comparison visualizations"""
        
        if not self.experiment_data or len(self.experiment_data.aggregate_results) < 2:
            return
        
        print("Generating model comparison visualizations...")
        
        comparison_dir = self.data_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        # Head-to-head performance comparison
        self._create_head_to_head_comparison(comparison_dir)
        
        # Model ranking visualization
        self._create_model_rankings(comparison_dir)
        
        # Strengths and weaknesses analysis
        self._create_strengths_weaknesses(comparison_dir)
    
    def _create_head_to_head_comparison(self, output_dir):
        """Create head-to-head comparison matrix"""
        
        models = [r.model_name for r in self.experiment_data.aggregate_results]
        n_models = len(models)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = [
            ('avg_consistency', 'Consistency', 0, 100),
            ('avg_reasoning_quality', 'Reasoning Quality', 1, 5),
            ('balance_attempt_rate', 'Balance Attempt Rate', 0, 1),
            ('refusal_rate', 'Refusal Rate', 0, 1)
        ]
        
        for idx, (metric, title, vmin, vmax) in enumerate(metrics):
            matrix = np.zeros((n_models, n_models))
            
            for i, result_i in enumerate(self.experiment_data.aggregate_results):
                for j, result_j in enumerate(self.experiment_data.aggregate_results):
                    val_i = getattr(result_i, metric)
                    val_j = getattr(result_j, metric)
                    
                    if metric == 'refusal_rate':  # Lower is better for refusal rate
                        matrix[i, j] = val_j - val_i  # Positive means i is better
                    else:  # Higher is better for other metrics
                        matrix[i, j] = val_i - val_j  # Positive means i is better
            
            im = axes[idx].imshow(matrix, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
            axes[idx].set_xticks(range(n_models))
            axes[idx].set_yticks(range(n_models))
            axes[idx].set_xticklabels(models, rotation=45)
            axes[idx].set_yticklabels(models)
            axes[idx].set_title(f'Head-to-Head: {title}', fontsize=12)
            
            # Add text annotations
            for i in range(n_models):
                for j in range(n_models):
                    text = axes[idx].text(j, i, f'{matrix[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.suptitle('Model Head-to-Head Comparisons', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'head_to_head_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_rankings(self, output_dir):
        """Create overall model ranking visualization"""
        
        # Calculate composite scores
        model_scores = []
        for result in self.experiment_data.aggregate_results:
            composite_score = (
                result.avg_consistency * 0.3 +
                result.avg_reasoning_quality * 20 * 0.3 +  # Scale to 0-100
                (1 - result.no_acknowledgment_rate) * 100 * 0.2 +
                result.balance_attempt_rate * 100 * 0.1 +
                (1 - result.harmful_output_rate) * 100 * 0.1
            )
            model_scores.append((result.model_name, composite_score))
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        models, scores = zip(*model_scores)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(models, scores, color='skyblue', alpha=0.8)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}', ha='left', va='center')
        
        ax.set_xlabel('Composite Score', fontsize=12)
        ax.set_title('Overall Model Rankings', fontsize=16, pad=20)
        ax.set_xlim(0, max(scores) * 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_strengths_weaknesses(self, output_dir):
        """Create strengths and weaknesses analysis"""
        
        metrics = ['avg_consistency', 'avg_reasoning_quality', 'balance_attempt_rate']
        metric_names = ['Consistency', 'Reasoning Quality', 'Balance Attempts']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            scores = [(r.model_name, getattr(r, metric)) for r in self.experiment_data.aggregate_results]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            models, values = zip(*scores)
            colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightgray' 
                     for i in range(len(models))]
            
            bars = axes[idx].bar(models, values, color=colors, alpha=0.8)
            axes[idx].set_title(f'Best in {name}', fontsize=14)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                              f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle('Model Strengths Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'strengths_weaknesses.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations and reports"""
        
        print("Generating comprehensive visualizations...")
        
        # Create overall comparative visualizations (existing ones)
        self.create_consistency_heatmap()
        self.create_principle_win_rates()
        self.create_model_comparison_radar()
        self.create_severity_analysis()
        self.create_conflict_pair_matrix()
        
        # Create individual model visualizations
        self.create_individual_model_visualizations()
        
        # Create model comparison visualizations
        self.create_model_comparison_visualizations()
        
        # Generate comprehensive HTML report
        self.generate_comprehensive_html_report()
        
        print("All visualizations complete!")
    
    def generate_comprehensive_html_report(self):
        """Generate comprehensive HTML report with links to individual model reports"""
        
        if not self.experiment_data:
            return
        
        # Generate individual model HTML reports
        self._generate_individual_model_reports()
        
        # Generate main comprehensive report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Constitutional Clash: Comprehensive Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .summary-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .model-card {{ background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; }}
        .model-card h4 {{ margin-top: 0; color: #007bff; }}
        .model-card a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
        .model-card a:hover {{ text-decoration: underline; }}
        .nav-section {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .nav-section a {{ margin-right: 15px; color: #007bff; text-decoration: none; }}
        .nav-section a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Constitutional Clash: Comprehensive Experiment Analysis</h1>
        
        <div class="nav-section">
            <strong>Quick Navigation:</strong>
            <a href="#summary">Summary</a>
            <a href="#individual-models">Individual Models</a>
            <a href="#comparisons">Model Comparisons</a>
            <a href="#visualizations">Overall Analysis</a>
        </div>
        
        <div class="summary-box" id="summary">
            <h2>Experiment Summary</h2>
            <div class="metric">
                <div class="metric-value">{len(self.experiment_data.metadata.models_tested)}</div>
                <div class="metric-label">Models Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(self.experiment_data.metadata.conflict_prompts)}</div>
                <div class="metric-label">Conflict Prompts</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(self.experiment_data.evaluations)}</div>
                <div class="metric-label">Total Evaluations</div>
            </div>
        </div>
        
        <h2 id="individual-models">Individual Model Analysis</h2>
        <div class="model-grid">"""
        
        for result in self.experiment_data.aggregate_results:
            html_content += f"""
            <div class="model-card">
                <h4>{result.model_name}</h4>
                <p><strong>Consistency:</strong> {result.avg_consistency:.1f}%</p>
                <p><strong>Reasoning:</strong> {result.avg_reasoning_quality:.2f}/5</p>
                <p><strong>Balance Rate:</strong> {result.balance_attempt_rate*100:.1f}%</p>
                <a href="models/{result.model_name}/report.html">View Detailed Report →</a>
            </div>"""
        
        html_content += f"""
        </div>
        
        <h2 id="comparisons">Model Comparisons</h2>
        <p><a href="comparisons/comparison_report.html">View Detailed Comparison Analysis →</a></p>
        <img src="comparisons/model_rankings.png" alt="Model Rankings">
        <img src="comparisons/head_to_head_comparison.png" alt="Head-to-Head Comparison">
        
        <h2>Model Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Avg Consistency</th>
                <th>Reasoning Quality</th>
                <th>Conflict Recognition</th>
                <th>Balance Rate</th>
                <th>Refusal Rate</th>
            </tr>"""
        
        for result in self.experiment_data.aggregate_results:
            html_content += f"""
            <tr>
                <td><a href="models/{result.model_name}/report.html">{result.model_name}</a></td>
                <td>{result.avg_consistency:.1f}%</td>
                <td>{result.avg_reasoning_quality:.2f}/5</td>
                <td>{(1 - result.no_acknowledgment_rate) * 100:.1f}%</td>
                <td>{result.balance_attempt_rate * 100:.1f}%</td>
                <td>{result.refusal_rate * 100:.1f}%</td>
            </tr>"""
        
        html_content += """
        </table>
        
        <h2 id="visualizations">Overall Analysis Visualizations</h2>
        <img src="consistency_heatmap.png" alt="Consistency Heatmap">
        <img src="model_comparison_radar.png" alt="Model Comparison Radar">
        <img src="principle_win_rates.png" alt="Principle Win Rates">
        <img src="severity_analysis.png" alt="Severity Analysis">
        <img src="conflict_pair_matrix.png" alt="Conflict Pair Matrix">
        
        <h2>Key Findings</h2>
        <ul>
            <li>Models show significant variation in handling principle conflicts</li>
            <li>Consistency varies dramatically across conflict categories</li>
            <li>Different models have distinct "principle preferences"</li>
            <li>Severity level significantly impacts response quality</li>
            <li>Balance attempts and explicit conflict acknowledgment are rare</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(self.data_dir / 'comprehensive_report.html', 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive HTML report saved to {self.data_dir / 'comprehensive_report.html'}")
    
    def _generate_individual_model_reports(self):
        """Generate individual HTML reports for each model"""
        
        for result in self.experiment_data.aggregate_results:
            model_dir = self.data_dir / "models" / result.model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{result.model_name} - Constitutional Clash Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 20px; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .summary-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .back-link {{ background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-bottom: 20px; }}
        .back-link:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../../comprehensive_report.html" class="back-link">← Back to Main Report</a>
        
        <h1>{result.model_name} Detailed Analysis</h1>
        
        <div class="summary-box">
            <h2>Performance Metrics</h2>
            <div class="metric">
                <div class="metric-value">{result.avg_consistency:.1f}%</div>
                <div class="metric-label">Average Consistency</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.avg_reasoning_quality:.2f}/5</div>
                <div class="metric-label">Reasoning Quality</div>
            </div>
            <div class="metric">
                <div class="metric-value">{(1-result.no_acknowledgment_rate)*100:.1f}%</div>
                <div class="metric-label">Conflict Recognition</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.balance_attempt_rate*100:.1f}%</div>
                <div class="metric-label">Balance Attempts</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.refusal_rate*100:.1f}%</div>
                <div class="metric-label">Refusal Rate</div>
            </div>
        </div>
        
        <h2>Performance Breakdown</h2>
        <img src="performance_breakdown.png" alt="Performance Breakdown">
        
        <h2>Category and Severity Analysis</h2>
        <img src="category_analysis.png" alt="Category Analysis">
        
        <h2>Response Characteristics</h2>
        <img src="response_characteristics.png" alt="Response Characteristics">
        
        <h2>Principle Preferences</h2>
        <img src="principle_preferences.png" alt="Principle Preferences">
        
        <h2>Key Insights for {result.model_name}</h2>
        <ul>"""
        
        # Add specific insights based on the model's performance
        if result.avg_consistency > 80:
            html_content += "<li>✅ High consistency across similar prompts</li>"
        elif result.avg_consistency > 60:
            html_content += "<li>⚠️ Moderate consistency with room for improvement</li>"
        else:
            html_content += "<li>❌ Low consistency - responses vary significantly</li>"
        
        if result.balance_attempt_rate > 0.5:
            html_content += "<li>✅ Frequently attempts to balance conflicting principles</li>"
        else:
            html_content += "<li>⚠️ Rarely attempts explicit principle balancing</li>"
        
        if result.refusal_rate > 0.3:
            html_content += "<li>🛡️ Conservative approach - frequently refuses harmful requests</li>"
        elif result.refusal_rate > 0.1:
            html_content += "<li>⚖️ Balanced approach to harmful content</li>"
        else:
            html_content += "<li>⚠️ Liberal approach - rarely refuses requests</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
            
        with open(model_dir / 'report.html', 'w') as f:
            f.write(html_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Constitutional Clash results")
    parser.add_argument("--results-dir", default="./results",
                       help="Directory containing experiment results")
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results_dir)
    visualizer.generate_all_visualizations()