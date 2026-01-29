#!/usr/bin/env python3
"""
Ablation Study Script for Prompt Optimization
Performs greedy depth-first ablation with perfect-score pruning on PROMPT_BASE.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import existing evaluation components
from evaluate_plans import PlanValidator, SPLCalculator, FailureType
from world_manager import WorldManager

# Import Ollama client
try:
    from ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None

# Import Google GenAI client
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

# Import PROMPT_BASE from nl_control
from nl_control import PROMPT_BASE


@dataclass
class AblationResult:
    """Result of a single ablation test."""
    iteration: int
    removed_section: str
    success: bool
    spl: Optional[float]
    input_tokens: int
    output_tokens: int
    time_seconds: float
    total_score: float
    failure_reason: str = ""


@dataclass
class PromptSection:
    """Represents a section of the prompt."""
    name: str
    content: str
    start_line: int
    end_line: int


class PromptSectionParser:
    """Parse PROMPT_BASE into logical sections based on whitespace/blank lines."""
    
    def __init__(self, prompt_text: str = None):
        self.prompt_text = prompt_text or PROMPT_BASE
        self.prompt_lines = self.prompt_text.split('\n')
    
    def parse_sections(self) -> List[PromptSection]:
        """Split prompt into sections based on blank lines and logical groupings."""
        sections = []
        current_section_lines = []
        current_section_start = 0
        section_number = 0
        
        for i, line in enumerate(self.prompt_lines):
            line_stripped = line.strip()
            
            # Check if this is a blank line (section separator)
            if not line_stripped:
                # If we have content in current section, save it
                if current_section_lines:
                    section_name = self._generate_section_name(current_section_lines, section_number)
                    section = PromptSection(
                        name=section_name,
                        content='\n'.join(current_section_lines),
                        start_line=current_section_start,
                        end_line=i - 1
                    )
                    sections.append(section)
                    section_number += 1
                
                # Reset for next section
                current_section_lines = []
                current_section_start = i + 1
            else:
                # Add line to current section
                current_section_lines.append(line)
        
        # Handle the last section (if any)
        if current_section_lines:
            section_name = self._generate_section_name(current_section_lines, section_number)
            section = PromptSection(
                name=section_name,
                content='\n'.join(current_section_lines),
                start_line=current_section_start,
                end_line=len(self.prompt_lines) - 1
            )
            sections.append(section)
        
        return sections
    
    def _generate_section_name(self, section_lines: List[str], section_number: int) -> str:
        """Generate a meaningful name for a section based on its content."""
        # Get the first non-empty line as the basis for the name
        first_line = ""
        for line in section_lines:
            if line.strip():
                first_line = line.strip()
                break
        
        if not first_line:
            return f"section_{section_number}"
        
        # Extract key words from the first line
        words = first_line.split()
        if len(words) >= 2:
            # Take first two meaningful words
            key_words = []
            for word in words[:3]:  # Look at first 3 words
                clean_word = word.strip('.,:;!?').lower()
                if clean_word and len(clean_word) > 2:  # Skip short words
                    key_words.append(clean_word)
                if len(key_words) >= 2:
                    break
            
            if key_words:
                return '_'.join(key_words)
        
        # Fallback: use first word or section number
        if words:
            clean_word = words[0].strip('.,:;!?').lower()
            if clean_word and len(clean_word) > 2:
                return clean_word
        
        return f"section_{section_number}"
    
    def create_prompt_without_section(self, sections: List[PromptSection], 
                                    removed_section_name: str) -> str:
        """Create a prompt with a specific section removed."""
        remaining_lines = []
        
        for i, line in enumerate(self.prompt_lines):
            # Check if this line belongs to the removed section
            belongs_to_removed = False
            for section in sections:
                if section.name == removed_section_name:
                    if section.start_line <= i <= section.end_line:
                        belongs_to_removed = True
                        break
            
            if not belongs_to_removed:
                remaining_lines.append(line)
        
        # Clean up any double blank lines that might be created
        cleaned_lines = []
        prev_was_empty = False
        for line in remaining_lines:
            is_empty = not line.strip()
            if is_empty and prev_was_empty:
                continue  # Skip consecutive empty lines
            cleaned_lines.append(line)
            prev_was_empty = is_empty
        
        return '\n'.join(cleaned_lines)


class LLMPlanGenerator:
    """Generate plans using LLM with custom prompts."""
    
    def __init__(self, model: str = "gpt-oss:120b"):
        self.model = model
        self.ollama_client = None
        self.gemini_client = None
        
        # Initialize Ollama for gpt-oss models
        if model.startswith("gpt-oss") and OLLAMA_AVAILABLE:
            try:
                self.ollama_client = OllamaClient()
                self.ollama_client.model = model
            except Exception as e:
                print(f"Warning: Failed to initialize Ollama client: {e}")
                self.ollama_client = None
        elif not OLLAMA_AVAILABLE and model.startswith("gpt-oss"):
            print("Warning: Ollama client not available")
        
        # Initialize Gemini for gemini models
            if model.startswith("gemini") and GENAI_AVAILABLE:
                # Accept either GEMINI_API_KEY or GOOGLE_API_KEY
                gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                print(f"Debug: GEMINI/GOOGLE API key found: {bool(gemini_api_key)}")
                if gemini_api_key:
                    try:
                        self.gemini_client = genai.Client(api_key=gemini_api_key)
                        print(f"✓ Gemini client initialized successfully")
                    except Exception as e:
                        print(f"Warning: Failed to initialize Gemini client: {e}")
                        self.gemini_client = None
                else:
                    print("Error: GEMINI_API_KEY or GOOGLE_API_KEY not found. Please set the environment variable.")
                    print("Available env vars:", [k for k in os.environ.keys() if 'GEMINI' in k.upper() or 'GOOGLE' in k.upper()])
                    self.gemini_client = None
        elif not GENAI_AVAILABLE and model.startswith("gemini"):
            print("Error: Google GenAI not available. Please install google-genai package.")
            self.gemini_client = None
    
    def generate_plan_with_prompt(self, prompt: str, task: str, 
                                world_config: Dict[str, Any]) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        """Generate a plan using custom prompt and track metrics."""
        # Check if we have a valid client
        if not self.ollama_client and not self.gemini_client:
            return None, {"error": f"No LLM client available for model {self.model}"}
        
        # Create current state JSON
        current_state_json = {
            "robot_state": "Stand",
            "standing": True,
            "robot_cell": [4, 4],  # Starting position for World 12
            "facing": "N",
            "arm": "stowed",
            "held_object": None
        }
        
        # Format the prompt
        formatted_prompt = prompt.format(
            current_state_json=json.dumps(current_state_json, indent=2),
            world_model_json=json.dumps(world_config, indent=2),
            command=task
        )
        
        # Track metrics
        start_time = time.time()
        
        try:
            # Estimate input tokens (rough approximation: chars / 4)
            input_tokens = len(formatted_prompt) // 4
            
            # Generate response based on model type
            if self.model.startswith("gpt-oss") and self.ollama_client:
                response = self.ollama_client.generate(
                    prompt=formatted_prompt,
                    options={
                        "temperature": 0.1,
                        "num_predict": 8192
                    }
                )
                
                if response and "response" in response:
                    result = response["response"].strip()
                else:
                    return None, {"error": "No response from Ollama", "time_seconds": time.time() - start_time}
                    
            elif self.model.startswith("gemini") and self.gemini_client:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=formatted_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    )
                )
                result = response.text.strip()
            else:
                return None, {"error": f"Unsupported model: {self.model}"}
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Parse actions from response
            actions = []
            for line in result.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        actions.append(line)
            
            # Estimate output tokens
            output_tokens = len(result) // 4
            
            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "time_seconds": generation_time,
                "success": True
            }
            
            return actions, metrics
                
        except Exception as e:
            end_time = time.time()
            return None, {
                "error": str(e),
                "time_seconds": end_time - start_time,
                "input_tokens": len(formatted_prompt) // 4,
                "output_tokens": 0
            }


class AblationEvaluator:
    """Evaluate plans using existing evaluation framework."""
    
    def __init__(self, world_config: Dict[str, Any], ground_truth: Dict[str, Any], model: str = "gpt-oss:120b"):
        self.validator = PlanValidator(world_config, ground_truth)
        self.spl_calculator = SPLCalculator()
        self.model = model
    
    def evaluate_plan_result(self, actions: List[str], plan_id: str = "ablation_test") -> Dict[str, Any]:
        """Evaluate a plan and return results."""
        if not actions:
            return {
                "success": False,
                "spl": None,
                "failure_type": "rejected",
                "failure_reason": "Empty plan"
            }
        
        # Create plan data in expected format
        plan_data = {
            "id": plan_id,
            "llm_model": self.model,
            "world_id": "12",
            "task": "pick up the maze package and drop it off",
            "plan": actions
        }
        
        # Validate the plan
        result = self.validator.validate_plan(plan_data)
        
        return {
            "success": result.success,
            "spl": result.spl,
            "failure_type": result.failure_type.value if result.failure_type else None,
            "failure_reason": result.failure_reason,
            "path_length": result.path_length,
            "optimal_path_length": result.optimal_path_length
        }


class ScoringCalculator:
    """Calculate normalized scores for ablation results."""
    
    def __init__(self):
        self.max_tokens_observed = 0
        self.max_time_observed = 0
    
    def calculate_normalized_score(self, success: bool, spl: Optional[float], 
                                 input_tokens: int, output_tokens: int, 
                                 time_seconds: float) -> float:
        """Calculate normalized total score."""
        # Update observed maximums
        total_tokens = input_tokens + output_tokens
        self.max_tokens_observed = max(self.max_tokens_observed, total_tokens)
        self.max_time_observed = max(self.max_time_observed, time_seconds)
        
        # Normalize scores (0-1 scale)
        success_score = 1.0 if success else 0.0
        spl_score = spl if spl is not None else 0.0
        
        # Token score (lower is better)
        token_score = 1.0 - (total_tokens / max(self.max_tokens_observed, 1))
        
        # Time score (lower is better)
        time_score = 1.0 - (time_seconds / max(self.max_time_observed, 1))
        
        # Weighted total score
        total_score = (success_score * 0.4) + (spl_score * 0.4) + (token_score * 0.1) + (time_score * 0.1)
        
        return total_score


class AblationStudy:
    """Main ablation study orchestrator with DFS aggressive pruning."""
    
    def __init__(self, world_id: str = "12", task: str = "pick_maze_package_dropoff", 
                 model: str = "gpt-oss:120b", prompt_text: str = None):
        self.world_id = world_id
        self.task = task
        self.model = model
        self.prompt_text = prompt_text or PROMPT_BASE
        
        # Load world configuration
        world_manager = WorldManager()
        self.world_config = world_manager.get_world_config(world_id)
        if not self.world_config:
            raise ValueError(f"Could not load world {world_id}")
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Initialize components
        self.parser = PromptSectionParser(self.prompt_text)
        self.llm_generator = LLMPlanGenerator(model)
        self.evaluator = AblationEvaluator(self.world_config, self.ground_truth, model)
        self.scorer = ScoringCalculator()
        
        # DFS state tracking
        self.sections = []
        self.section_names = []
        self.seen_combinations = set()  # Cache for bitmask combinations
        self.best_subset = None
        self.best_score = 0.0
        self.results: List[AblationResult] = []
        self.evaluation_cache = {}  # Cache evaluation results
    
    def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth data."""
        ground_truth_file = "evaluation_files/ground_truth.json"
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
        
        world_key = f"world_{self.world_id}"
        if world_key not in data:
            raise ValueError(f"No ground truth data for {world_key}")
        
        if self.task not in data[world_key]:
            raise ValueError(f"No ground truth data for task '{self.task}' in {world_key}")
        
        return data[world_key][self.task]
    
    def run_ablation_iteration(self, iteration: int, removed_section: str) -> AblationResult:
        """Run a single ablation test iteration."""
        print(f"  Testing iteration {iteration}: removing '{removed_section}'...")
        
        # For baseline test, use full prompt
        if removed_section == "None (Baseline)":
            test_prompt = PROMPT_BASE
        else:
            # Create prompt without the section
            sections = self.parser.parse_sections()
            test_prompt = self.parser.create_prompt_without_section(sections, removed_section)
        
        # Generate plan
        actions, llm_metrics = self.llm_generator.generate_plan_with_prompt(
            test_prompt, "pick up the maze package and drop it off", self.world_config
        )
        
        # Debug output
        if "error" in llm_metrics:
            print(f"    LLM Error: {llm_metrics['error']}")
        else:
            print(f"    Generated {len(actions) if actions else 0} actions")
            if actions:
                print(f"    First action: {actions[0]}")
                print(f"    Last action: {actions[-1]}")
        
        # Evaluate plan
        eval_result = self.evaluator.evaluate_plan_result(actions)
        
        # Calculate score
        total_score = self.scorer.calculate_normalized_score(
            eval_result["success"],
            eval_result["spl"],
            llm_metrics.get("input_tokens", 0),
            llm_metrics.get("output_tokens", 0),
            llm_metrics.get("time_seconds", 0)
        )
        
        # Create result
        result = AblationResult(
            iteration=iteration,
            removed_section=removed_section,
            success=eval_result["success"],
            spl=eval_result["spl"],
            input_tokens=llm_metrics.get("input_tokens", 0),
            output_tokens=llm_metrics.get("output_tokens", 0),
            time_seconds=llm_metrics.get("time_seconds", 0),
            total_score=total_score,
            failure_reason=eval_result.get("failure_reason", "")
        )
        
        return result
    
    def _create_prompt_from_subset(self, subset: set) -> str:
        """Create prompt from a subset of sections."""
        if not subset:  # Empty subset means full prompt
            return self.prompt_text
        
        # Get all sections from the original prompt
        sections = self.parser.parse_sections()
        removed_sections = set(self.section_names) - subset
        
        # Create a new parser for the current prompt state
        current_prompt = self.prompt_text
        for removed_section in removed_sections:
            # Create a temporary parser for the current prompt
            temp_parser = PromptSectionParser(current_prompt)
            current_prompt = temp_parser.create_prompt_without_section(
                temp_parser.parse_sections(), removed_section
            )
        
        return current_prompt
    
    def _subset_to_bitmask(self, subset: set) -> int:
        """Convert subset to bitmask for caching."""
        bitmask = 0
        for i, section_name in enumerate(self.section_names):
            if section_name in subset:
                bitmask |= (1 << i)
        return bitmask
    
    def _bitmask_to_subset(self, bitmask: int) -> set:
        """Convert bitmask back to subset."""
        subset = set()
        for i, section_name in enumerate(self.section_names):
            if bitmask & (1 << i):
                subset.add(section_name)
        return subset
    
    def _evaluate_subset(self, subset: set) -> Tuple[bool, float, Dict[str, Any]]:
        """Evaluate a subset of sections and return (success, score, metrics)."""
        # Check cache first
        bitmask = self._subset_to_bitmask(subset)
        if bitmask in self.evaluation_cache:
            return self.evaluation_cache[bitmask]
        
        # Create prompt from subset
        prompt = self._create_prompt_from_subset(subset)
        
        # Generate plan
        actions, llm_metrics = self.llm_generator.generate_plan_with_prompt(
            prompt, "pick up the maze package and drop it off", self.world_config
        )
        
        # Evaluate plan
        eval_result = self.evaluator.evaluate_plan_result(actions)
        
        # Calculate score
        total_score = self.scorer.calculate_normalized_score(
            eval_result["success"],
            eval_result["spl"],
            llm_metrics.get("input_tokens", 0),
            llm_metrics.get("output_tokens", 0),
            llm_metrics.get("time_seconds", 0)
        )
        
        # Cache result
        result = (eval_result["success"], total_score, {
            "spl": eval_result["spl"],
            "input_tokens": llm_metrics.get("input_tokens", 0),
            "output_tokens": llm_metrics.get("output_tokens", 0),
            "time_seconds": llm_metrics.get("time_seconds", 0),
            "failure_reason": eval_result.get("failure_reason", "")
        })
        
        self.evaluation_cache[bitmask] = result
        return result
    
    def _order_sections_by_heuristic(self, current_subset: set) -> List[str]:
        """Order sections by heuristic (least harmful first)."""
        # For now, use simple ordering by section name
        # In practice, you could order by token count, importance, etc.
        available_sections = [s for s in self.section_names if s in current_subset]
        return sorted(available_sections)
    
    def _dfs_aggressive_pruning(self, current_subset: set, depth: int = 0) -> None:
        """DFS with aggressive pruning - only continue on 100% subsets."""
        indent = "  " * depth
        
        # Check if we've seen this combination
        bitmask = self._subset_to_bitmask(current_subset)
        if bitmask in self.seen_combinations:
            return
        self.seen_combinations.add(bitmask)
        
        # Bound check: if current subset is larger than best, prune
        if self.best_subset and len(current_subset) >= len(self.best_subset):
            return
        
        print(f"{indent}Testing subset of size {len(current_subset)}: {sorted(current_subset)}")
        
        # Evaluate current subset
        success, score, metrics = self._evaluate_subset(current_subset)
        
        # Record result
        result = AblationResult(
            iteration=len(self.results),
            removed_section=f"Subset: {sorted(current_subset)}",
            success=success,
            spl=metrics["spl"],
            input_tokens=metrics["input_tokens"],
            output_tokens=metrics["output_tokens"],
            time_seconds=metrics["time_seconds"],
            total_score=score,
            failure_reason=metrics["failure_reason"]
        )
        self.results.append(result)
        
        spl_str = f"{metrics['spl']:.3f}" if metrics['spl'] is not None else "N/A"
        print(f"{indent}  Result: Success={success}, SPL={spl_str}, Score={score:.3f}")
        
        # Check if this is a perfect score (100%)
        if success and metrics["spl"] == 1.0:
            print(f"{indent}  ✓ Perfect score achieved!")
            
            # Update best if this is smaller
            if not self.best_subset or len(current_subset) < len(self.best_subset):
                self.best_subset = current_subset.copy()
                self.best_score = score
                print(f"{indent}  🎯 New best subset found: {len(current_subset)} sections")
            
            # Continue DFS with children (remove one more section)
            improved = False
            for section_name in self._order_sections_by_heuristic(current_subset):
                child_subset = current_subset - {section_name}
                if child_subset:  # Don't go to empty subset
                    print(f"{indent}  → Trying to remove '{section_name}'...")
                    self._dfs_aggressive_pruning(child_subset, depth + 1)
                    improved = True
            
            if not improved:
                print(f"{indent}  🏁 Local minimum found: {sorted(current_subset)}")
        else:
            print(f"{indent}  ✗ Not perfect score, pruning this branch")
    
    def dfs_aggressive_pruning_ablation(self) -> List[AblationResult]:
        """Run DFS with aggressive pruning to find minimal perfect subset."""
        print("Starting DFS aggressive pruning ablation study...")
        
        # Parse sections
        self.sections = self.parser.parse_sections()
        self.section_names = [s.name for s in self.sections]
        
        print(f"Found {len(self.section_names)} sections: {self.section_names}")
        
        # Start with full set
        full_subset = set(self.section_names)
        print(f"\nStarting with full subset: {sorted(full_subset)}")
        
        # Run DFS
        self._dfs_aggressive_pruning(full_subset)
        
        print(f"\nDFS complete!")
        print(f"Best subset found: {len(self.best_subset)} sections" if self.best_subset else "No perfect subset found")
        if self.best_subset:
            print(f"Best sections: {sorted(self.best_subset)}")
            
            # Show the final prompt preview
            final_prompt = self._create_prompt_from_subset(self.best_subset)
            print(f"\nFinal prompt preview (first 3 lines):")
            for i, line in enumerate(final_prompt.split('\n')[:3]):
                print(f"  {i+1}: {line}")
            print(f"  ... (total length: {len(final_prompt)} chars)")
        
        return self.results
    
    def generate_markdown_report(self) -> str:
        """Generate markdown report with results table."""
        lines = []
        
        # Header
        lines.append("# Ablation Study Results")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Model:** {self.model}")
        lines.append(f"**World:** {self.world_id}")
        lines.append(f"**Task:** {self.task}")
        lines.append("")
        
        # Results table
        lines.append("## Results Table")
        lines.append("")
        lines.append("| Iteration | Removed Section | Success | SPL | Input Tokens | Output Tokens | Time (s) | Total Score |")
        lines.append("|-----------|----------------|---------|-----|--------------|---------------|----------|-------------|")
        
        for result in self.results:
            success_symbol = "✓" if result.success else "✗"
            spl_str = f"{result.spl:.3f}" if result.spl is not None else "N/A"
            
            lines.append(f"| {result.iteration} | {result.removed_section} | {success_symbol} | {spl_str} | {result.input_tokens} | {result.output_tokens} | {result.time_seconds:.2f} | {result.total_score:.3f} |")
        
        lines.append("")
        
        # Summary
        successful_results = [r for r in self.results if r.success]
        perfect_results = [r for r in successful_results if r.spl == 1.0]
        
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Tests:** {len(self.results)}")
        lines.append(f"- **Successful Plans:** {len(successful_results)}")
        lines.append(f"- **Perfect Score Plans (SPL=1.0):** {len(perfect_results)}")
        lines.append(f"- **Best Subset Size:** {len(self.best_subset) if self.best_subset else 'None found'}")
        lines.append(f"- **Best Subset Sections:** {', '.join(sorted(self.best_subset)) if self.best_subset else 'None'}")
        lines.append(f"- **Removed Sections:** {', '.join(sorted(set(self.section_names) - self.best_subset)) if self.best_subset else 'None'}")
        lines.append(f"- **Cache Efficiency:** {len(self.evaluation_cache)} unique evaluations")
        lines.append("")
        
        # Best result
        if self.results:
            best_result = max(self.results, key=lambda r: r.total_score)
            lines.append("## Best Result")
            lines.append("")
            lines.append(f"- **Iteration:** {best_result.iteration}")
            lines.append(f"- **Removed Section:** {best_result.removed_section}")
            lines.append(f"- **Success:** {best_result.success}")
            spl_str = f"{best_result.spl:.3f}" if best_result.spl is not None else "N/A"
            lines.append(f"- **SPL:** {spl_str}")
            lines.append(f"- **Total Score:** {best_result.total_score:.3f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_winner_json(self) -> Dict[str, Any]:
        """Generate winner JSON with best prompt configuration."""
        if not self.results:
            return {}
        
        # Find best result
        best_result = max(self.results, key=lambda r: r.total_score)
        
        # Get final prompt from best subset
        if self.best_subset:
            final_prompt = self._create_prompt_from_subset(self.best_subset)
            removed_sections = set(self.section_names) - self.best_subset
            
            # Debug: Show prompt length difference
            original_length = len(PROMPT_BASE)
            final_length = len(final_prompt)
            print(f"Debug: Original prompt length: {original_length}")
            print(f"Debug: Final prompt length: {final_length}")
            print(f"Debug: Removed {len(removed_sections)} sections: {sorted(removed_sections)}")
        else:
            final_prompt = PROMPT_BASE
            removed_sections = []
        
        return {
            "winning_prompt": final_prompt,
            "performance": {
                "success": best_result.success,
                "spl": best_result.spl,
                "input_tokens": best_result.input_tokens,
                "output_tokens": best_result.output_tokens,
                "time_seconds": best_result.time_seconds,
                "total_score": best_result.total_score
            },
            "best_subset": sorted(self.best_subset) if self.best_subset else [],
            "removed_sections": sorted(removed_sections),
            "ablation_path": [r.removed_section for r in self.results],
            "model": self.model,
            "world_id": self.world_id,
            "task": self.task,
            "total_evaluations": len(self.results),
            "cache_hits": len(self.evaluation_cache)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation study on prompt optimization")
    parser.add_argument("--world", default="12", help="World ID (default: 12)")
    parser.add_argument("--task", default="pick_maze_package_dropoff", 
                       help="Task key (default: pick_maze_package_dropoff)")
    parser.add_argument("--model", default="gpt-oss:120b", 
                       help="LLM model (default: gpt-oss:120b). Options: gpt-oss:120b, gemini-2.5-pro")
    parser.add_argument("--prompt-file", 
                       help="Custom prompt file to use instead of PROMPT_BASE")
    parser.add_argument("--output-dir", default="results", 
                       help="Output directory (default: results)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load custom prompt if specified
    prompt_text = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            print(f"Loaded custom prompt from: {args.prompt_file}")
        except Exception as e:
            print(f"Error loading prompt file: {e}")
            sys.exit(1)
    
    # Initialize ablation study
    try:
        study = AblationStudy(
            world_id=args.world,
            task=args.task,
            model=args.model,
            prompt_text=prompt_text
        )
    except Exception as e:
        print(f"Error initializing ablation study: {e}")
        sys.exit(1)
    
    # Run ablation study
    print("="*60)
    print("ABLATION STUDY FOR PROMPT OPTIMIZATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"World: {args.world}")
    print(f"Task: {args.task}")
    print("="*60)
    
    results = study.dfs_aggressive_pruning_ablation()
    
    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Markdown report
    markdown_report = study.generate_markdown_report()
    markdown_file = f"{args.output_dir}/ablation_study_{timestamp}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"\nMarkdown report saved to: {markdown_file}")
    
    # Winner JSON
    winner_json = study.generate_winner_json()
    json_file = f"{args.output_dir}/ablation_winner_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(winner_json, f, indent=2)
    print(f"Winner JSON saved to: {json_file}")
    
    print("\nAblation study complete!")


if __name__ == "__main__":
    main()
