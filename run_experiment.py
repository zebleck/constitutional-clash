"""Main experiment runner for Constitutional Clash"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track

from data_structures import (
    ExperimentRun, ExperimentData, ExperimentConfig,
    ConflictPrompt, ModelResponse, EvaluationScore
)
from constitution import (
    get_custom_constitution, get_basic_constitution
)
from prompt_generator import PromptGenerator, get_manual_prompts
from llm_client import MultiModelRunner
from evaluator import ResponseEvaluator, ResultsAnalyzer

console = Console()

class ConstitutionalClashExperiment:
    """Main experiment orchestrator"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.constitution = self._load_constitution()
        self.prompts = []
        self.responses = {}
        self.evaluations = []
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        constitution_name = config.constitution_option
        models_str = "_".join(config.models)
        
        run_name = f"{timestamp}_{constitution_name}_{models_str}"
        self.results_dir = Path(config.output_dir) / run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_constitution(self):
        """Load constitution based on config"""
        if self.config.constitution_option == "basic":
            return get_basic_constitution()
        else:  # custom
            return get_custom_constitution()
    
    async def generate_prompts(self, use_manual: bool = False, prompts_file: Optional[str] = None) -> List[ConflictPrompt]:
        """Generate or load conflict prompts"""
        
        if prompts_file:
            console.print(f"[cyan]Loading prompts from {prompts_file}...[/cyan]")
            self.prompts = PromptGenerator.load_prompts(prompts_file)
        elif use_manual:
            console.print("[yellow]Using manual prompts...[/yellow]")
            self.prompts = get_manual_prompts(
                self.constitution, 
                num_per_category=self.config.num_prompts_per_category
            )
        else:
            console.print("[cyan]Generating prompts with LLM...[/cyan]")
            generator = PromptGenerator("gpt-4o-mini")
            self.prompts = await generator.generate_prompts(
                self.constitution,
                num_per_category=self.config.num_prompts_per_category,
                include_variations=True
            )
            
            # Save prompts
            generator.save_prompts(self.prompts, self.results_dir / "prompts.json")
        
        console.print(f"[green]✓ Loaded/Generated {len(self.prompts)} prompts[/green]")
        return self.prompts
    
    async def run_models(self, resolution_mechanism: str = "baseline") -> Dict[str, List[ModelResponse]]:
        """Run all models on prompts"""
        
        console.print(f"\n[cyan]Testing models with {resolution_mechanism} mechanism...[/cyan]")
        
        runner = MultiModelRunner(self.config.models, self.config)
        self.responses = await runner.run_all_models(
            self.prompts,
            self.constitution,
            resolution_mechanism
        )
        
        # Save responses
        all_responses = []
        for model_responses in self.responses.values():
            all_responses.extend(model_responses)
        
        with open(self.results_dir / f"responses_{resolution_mechanism}.json", 'w') as f:
            json.dump(
                [r.model_dump(mode='json') for r in all_responses],
                f, indent=2, default=str
            )
        
        console.print(f"[green]✓ Collected {len(all_responses)} responses[/green]")
        return self.responses
    
    async def evaluate_responses(self) -> List[EvaluationScore]:
        """Evaluate all model responses"""
        
        console.print("\n[cyan]Evaluating responses...[/cyan]")
        
        evaluator = ResponseEvaluator(evaluation_model="gpt-4")
        
        # Create prompt lookup
        prompt_dict = {p.prompt_id: p for p in self.prompts}
        
        all_evaluations = []
        for model_name, model_responses in self.responses.items():
            console.print(f"  Evaluating {model_name}...")
            
            evals = await evaluator.batch_evaluate(
                model_responses,
                prompt_dict,
                self.constitution,
                show_progress=True
            )
            all_evaluations.extend(evals)
        
        self.evaluations = all_evaluations
        
        # Save evaluations
        with open(self.results_dir / "evaluations.json", 'w') as f:
            json.dump(
                [e.model_dump(mode='json') for e in all_evaluations],
                f, indent=2, default=str
            )
        
        console.print(f"[green]✓ Completed {len(all_evaluations)} evaluations[/green]")
        return all_evaluations
    
    def analyze_results(self) -> ExperimentData:
        """Analyze results and generate statistics"""
        
        console.print("\n[cyan]Analyzing results...[/cyan]")
        
        analyzer = ResultsAnalyzer()
        prompt_dict = {p.prompt_id: p for p in self.prompts}
        
        # Generate run metadata
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_run = ExperimentRun(
            run_id=run_id,
            constitution=self.constitution,
            conflict_prompts=self.prompts,
            models_tested=self.config.models,
            resolution_mechanism="baseline",  # TODO: Support multiple
            parameters={
                "temperature": 0.0,
                "max_tokens": 1000
            }
        )
        
        # Analyze each model
        aggregate_results = []
        for model_name in self.config.models:
            model_evals = [e for e in self.evaluations if e.model_name == model_name]
            
            if model_evals:
                result = analyzer.analyze_model_results(
                    model_name,
                    model_evals,
                    prompt_dict,
                    self.constitution,
                    run_id
                )
                aggregate_results.append(result)
        
        # Analyze conflict pairs
        conflict_pairs = analyzer.analyze_conflict_pairs(
            self.evaluations,
            prompt_dict
        )
        
        # Create experiment data
        all_responses = []
        for model_responses in self.responses.values():
            all_responses.extend(model_responses)
        
        experiment_data = ExperimentData(
            metadata=experiment_run,
            responses=all_responses,
            evaluations=self.evaluations,
            aggregate_results=aggregate_results,
            conflict_pairs=conflict_pairs
        )
        
        # Save complete results
        experiment_data.save_json(str(self.results_dir / "experiment_data.json"))
        
        # Print summary table
        self._print_summary(aggregate_results)
        
        return experiment_data
    
    def _print_summary(self, results: List):
        """Print summary table of results"""
        
        table = Table(title="Constitutional Clash Results Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Consistency", style="green")
        table.add_column("Reasoning", style="yellow")
        table.add_column("Conflict Ack.", style="magenta")
        table.add_column("Balance Rate", style="blue")
        table.add_column("Refusal Rate", style="red")
        
        for result in results:
            table.add_row(
                result.model_name,
                f"{result.avg_consistency:.1f}%",
                f"{result.avg_reasoning_quality:.2f}/5",
                f"{100 - result.no_acknowledgment_rate*100:.1f}%",
                f"{result.balance_attempt_rate*100:.1f}%",
                f"{result.refusal_rate*100:.1f}%"
            )
        
        console.print("\n")
        console.print(table)
        
        # Print conflict pair analysis
        console.print("\n[bold]Most Challenging Conflict Pairs:[/bold]")
        conflict_table = Table()
        conflict_table.add_column("Principle 1")
        conflict_table.add_column("Principle 2")
        conflict_table.add_column("Avg Consistency")
        conflict_table.add_column("Typical Winner")
        
        # Sort by consistency (lower = more challenging)
        if hasattr(self, 'conflict_pairs'):
            sorted_pairs = sorted(self.conflict_pairs, key=lambda x: x.avg_consistency)[:5]
            for pair in sorted_pairs:
                conflict_table.add_row(
                    pair.principle_1_id,
                    pair.principle_2_id,
                    f"{pair.avg_consistency:.1f}%",
                    pair.typical_resolution or "Mixed"
                )
            console.print(conflict_table)
    
    async def run_full_experiment(self, use_manual_prompts: bool = False, prompts_file: Optional[str] = None):
        """Run the complete experiment pipeline"""
        
        console.print("[bold green]Starting Constitutional Clash Experiment[/bold green]\n")
        
        # Generate prompts
        await self.generate_prompts(use_manual=use_manual_prompts, prompts_file=prompts_file)
        
        # Run models
        for mechanism in self.config.resolution_mechanisms:
            await self.run_models(mechanism)
        
        # Evaluate responses
        if self.config.auto_evaluate:
            await self.evaluate_responses()
            
            # Analyze results
            experiment_data = self.analyze_results()
            
            console.print(f"\n[green]✓ Experiment complete! Results saved to {self.results_dir}[/green]")
            return experiment_data
        else:
            console.print("[yellow]Skipping auto-evaluation. Run evaluator separately.[/yellow]")
            return None


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Run Constitutional Clash experiment")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "claude-sonnet-4"],
                       help="Models to test")
    parser.add_argument("--constitution", choices=["basic", "custom"],
                       default="custom", help="Constitution to use")
    parser.add_argument("--prompts-per-category", type=int, default=5,
                       help="Number of prompts per conflict category")
    parser.add_argument("--manual-prompts", action="store_true",
                       help="Use manual prompts instead of generating")
    parser.add_argument("--prompts-file", type=str,
                       help="Load prompts from existing JSON file (overrides --manual-prompts)")
    parser.add_argument("--output-dir", default="./results",
                       help="Output directory for results")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip automatic evaluation")
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        constitution_option=args.constitution,
        num_prompts_per_category=args.prompts_per_category,
        models=args.models,
        resolution_mechanisms=["baseline"],  # Can add more later
        auto_evaluate=not args.no_eval,
        output_dir=args.output_dir
    )
    
    # Run experiment
    experiment = ConstitutionalClashExperiment(config)
    await experiment.run_full_experiment(use_manual_prompts=args.manual_prompts, prompts_file=args.prompts_file)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise