#!/usr/bin/env python

import json
import openai
import os
import sys
from typing import List, Dict, Any, Tuple
import time

# Import prompts from the separate prompts file - REQUIRED
try:
    from prompts import get_all_prompts, get_prompt, get_prompt_names, add_prompt, remove_prompt
    PROMPTS_AVAILABLE = True
except ImportError:
    print("ERROR: prompts.py file not found!")
    print("This system requires prompts.py to function.")
    print("Please ensure prompts.py is in the same directory.")
    sys.exit(1)

class PromptTester:
    def __init__(self, prompt_names_to_test=None):
        """Initialize the prompt tester with OpenAI API.
        
        Args:
            prompt_names_to_test: List of prompt names to test, or None to test all available prompts
        """
        self.setup_openai()
        
        # Define the three test tasks with ground truth
        self.test_tasks = {
            "task1_simple": {
                "description": "Stand up and move your arm forward",
                "ground_truth": [
                    {"op": "stand", "height": 0.0},
                    {"op": "aim_hand", "x": 0.75, "y": 0.0, "z": 0.0}
                ],
                "expected_steps": 2,
                "complexity": "Simple (2 steps)"
            },
            "task2_medium": {
                "description": "Walk in a square",
                "ground_truth": [
                    {"op": "stand", "height": 0.2},
                    {"op": "move_relative", "x": 2.0, "y": 0.0, "yaw": 0.0, "timeout": 10.0},
                    {"op": "move_relative", "x": 0.0, "y": 0.0, "yaw": 1.5708, "timeout": 5.0},
                    {"op": "move_relative", "x": 2.0, "y": 0.0, "yaw": 0.0, "timeout": 10.0},
                    {"op": "move_relative", "x": 0.0, "y": 0.0, "yaw": 1.5708, "timeout": 5.0},
                    {"op": "move_relative", "x": 2.0, "y": 0.0, "yaw": 0.0, "timeout": 10.0},
                    {"op": "move_relative", "x": 0.0, "y": 0.0, "yaw": 1.5708, "timeout": 5.0},
                    {"op": "move_relative", "x": 2.0, "y": 0.0, "yaw": 0.0, "timeout": 10.0},
                    {"op": "move_relative", "x": 0.0, "y": 0.0, "yaw": 1.5708, "timeout": 5.0}
                ],
                "expected_steps": 9,
                "complexity": "Medium (9 steps)",
                "note": "Uses forward movement + rotation pattern to create square"
            },
            "task3_complex": {
                "description": "Look for the cup and bring it back",
                "ground_truth": [
                    {"op": "stand", "height": 0.0},
                    {"op": "aim_hand", "x": 0.75, "y": 0.0, "z": 0.0},
                    {"op": "detect_label", "label": "cup", "conf": 0.01, "image_source": "hand_color_image"},
                    {"op": "pick_from_pixel", "image_source": "hand_color_image", "x": 0, "y": 0, "top_down": True, "auto_walk": True, "timeout": 25.0}
                ],
                "expected_steps": 4,
                "complexity": "Complex (4 steps)"
            }
        }
        
        # Load prompts from the prompts file - REQUIRED
        if prompt_names_to_test:
            # Test only specified prompts
            self.prompts = {}
            for name in prompt_names_to_test:
                try:
                    prompt = get_prompt(name)
                    self.prompts[name] = prompt
                except ValueError as e:
                    print(f"Warning: {e}")
            print(f"Loaded {len(self.prompts)} specified prompts")
        else:
            # Test all available prompts
            self.prompts = get_all_prompts()
            print(f"Loaded {len(self.prompts)} available prompts")
    
    def setup_openai(self):
        """Setup OpenAI API."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in environment")
                print("LLM functionality will be disabled")
                self.use_llm = False
                return
            
            print("OpenAI API configured successfully")
            self.use_llm = True
        except Exception as e:
            print(f"Error setting up OpenAI: {e}")
            self.use_llm = False
    
    def add_custom_prompt(self, key: str, name: str, description: str, content: str):
        """Add a custom prompt for testing."""
        self.prompts[key] = {
            "name": name,
            "description": description,
            "content": content
        }
        print(f"Added custom prompt: {name}")
    
    def test_prompt_on_task(self, prompt_content: str, task_description: str, task_name: str) -> Dict[str, Any]:
        """Test a specific prompt on a specific task."""
        if not self.use_llm:
            return {"error": "OpenAI API not configured"}
        
        try:
            # Create the prompt
            full_prompt = f"""{prompt_content}

User instruction: '{task_description}'

Please convert this instruction to robot actions:"""
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Use GPT-5 API format
            full_input = f"""{prompt_content}

User instruction: '{task_description}'

Please convert this instruction to robot actions:"""
            
            response = client.responses.create(
                model="gpt-5",
                input=full_input,
                reasoning={"effort": "minimal"}
            )
            
            # Extract the response content
            llm_response = response.output_text.strip()
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = llm_response
            if llm_response.startswith('```'):
                # Remove markdown code blocks
                lines = llm_response.split('\n')
                if len(lines) >= 3:
                    # Skip first line (```json), last line (```), and join the middle
                    cleaned_response = '\n'.join(lines[1:-1])
            
            # Try to parse as JSON
            try:
                actions = json.loads(cleaned_response)
                if isinstance(actions, list):
                    return {
                        "success": True,
                        "actions": actions,
                        "raw_response": llm_response,
                        "cleaned_response": cleaned_response,
                        "num_actions": len(actions)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"LLM returned non-list: {cleaned_response}",
                        "raw_response": llm_response,
                        "cleaned_response": cleaned_response
                    }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"LLM response is not valid JSON: {cleaned_response}",
                    "raw_response": llm_response,
                    "cleaned_response": cleaned_response,
                    "json_error": str(e)
                }
                
        except Exception as e:
            return {"error": f"Error calling LLM: {e}"}
    
    def evaluate_plan_quality(self, generated_actions: List[Dict], ground_truth: List[Dict], task_name: str) -> Dict[str, Any]:
        """Evaluate the quality of a generated plan against ground truth."""
        if not generated_actions:
            return {"error": "No actions to evaluate"}
        
        # Basic metrics
        num_generated = len(generated_actions)
        num_expected = len(ground_truth)
        
        # Check if key actions are present
        key_actions = {
            "stand": False,
            "sit": False,
            "move_relative": False,
            "aim_hand": False,
            "detect_label": False,
            "pick_from_pixel": False
        }
        
        for action in generated_actions:
            op = action.get('op', '')
            if op in key_actions:
                key_actions[op] = True
        
        # Check for common issues
        issues = []
        if not key_actions["stand"]:
            issues.append("Missing stand action")
        # Note: sit action is not required for these tasks
        
        # Check for invalid operations
        invalid_ops = []
        for action in generated_actions:
            op = action.get('op', '')
            valid_ops = ['stand', 'sit', 'move_relative', 'aim_hand', 'detect_label', 'pick_from_pixel', 'gripper']
            if op not in valid_ops:
                invalid_ops.append(op)
        
        # Check for yaw values in degrees instead of radians
        yaw_issues = []
        for action in generated_actions:
            if action.get('op') == 'move_relative' and 'yaw' in action:
                yaw = action['yaw']
                if isinstance(yaw, (int, float)) and abs(yaw) > 6.28:  # Likely degrees
                    yaw_issues.append(f"Yaw value {yaw} appears to be in degrees (should be radians)")
        
        if yaw_issues:
            issues.extend(yaw_issues)
        
        if invalid_ops:
            issues.append(f"Invalid operations: {invalid_ops}")
        
        # Check parameter ranges based on natural_language_control.py limits
        range_issues = []
        for action in generated_actions:
            op = action.get('op', '')
            
            if op == 'stand' and 'height' in action:
                height = action['height']
                if isinstance(height, (int, float)) and (height < -0.2 or height > 0.5):
                    range_issues.append(f"Stand height {height} outside range [-0.2, 0.5]")
            
            elif op == 'move_relative':
                x = action.get('x', 0)
                y = action.get('y', 0)
                timeout = action.get('timeout', 10.0)
                
                if isinstance(x, (int, float)) and (x < -5.0 or x > 5.0):
                    range_issues.append(f"Move_relative x {x} outside range [-5.0, 5.0]")
                if isinstance(y, (int, float)) and (y < -3.0 or y > 3.0):
                    range_issues.append(f"Move_relative y {y} outside range [-3.0, 3.0]")
                if isinstance(timeout, (int, float)) and (timeout < 1.0 or timeout > 60.0):
                    range_issues.append(f"Move_relative timeout {timeout} outside range [1.0, 60.0]")
            
            elif op == 'aim_hand':
                x = action.get('x', 0)
                y = action.get('y', 0)
                z = action.get('z', 0)
                
                if isinstance(x, (int, float)) and (x < 0.60 or x > 0.90):
                    range_issues.append(f"Aim_hand x {x} outside range [0.60, 0.90]")
                if isinstance(y, (int, float)) and (y < -0.5 or y > 0.5):
                    range_issues.append(f"Aim_hand y {y} outside range [-0.5, 0.5]")
                if isinstance(z, (int, float)) and (z < -0.2 or z > 0.65):
                    range_issues.append(f"Aim_hand z {z} outside range [-0.2, 0.65]")
            
            elif op == 'pick_from_pixel' and 'timeout' in action:
                timeout = action['timeout']
                if isinstance(timeout, (int, float)) and (timeout < 5.0 or timeout > 60.0):
                    range_issues.append(f"Pick_from_pixel timeout {timeout} outside range [5.0, 60.0]")
        
        if range_issues:
            issues.extend(range_issues)
        
        # Calculate score based on ground truth similarity and parameter ranges
        score = 0
        max_score = 100
        
        # Check if actions match ground truth structure
        if num_generated == num_expected:
            score += 30  # Perfect step count match
        elif abs(num_generated - num_expected) <= 1:
            score += 20  # Close step count
        elif abs(num_generated - num_expected) <= 2:
            score += 10  # Reasonable step count
        elif abs(num_generated - num_expected) <= 3:
            score += 5   # Acceptable step count
        
        # Score for having required actions
        if key_actions["stand"]:
            score += 20
        if task_name == "task1_simple" and key_actions["aim_hand"]:
            score += 20
        elif task_name == "task2_medium" and key_actions["move_relative"]:
            score += 20
        elif task_name == "task3_complex" and (key_actions["aim_hand"] and key_actions["detect_label"] and key_actions["pick_from_pixel"]):
            score += 20
        
        # Bonus for parameters within correct ranges
        range_bonus = 0
        for action in generated_actions:
            op = action.get('op', '')
            
            if op == 'stand' and 'height' in action:
                height = action['height']
                if isinstance(height, (int, float)) and -0.2 <= height <= 0.5:
                    range_bonus += 5
            
            elif op == 'move_relative':
                x, y, timeout = action.get('x', 0), action.get('y', 0), action.get('timeout', 10.0)
                if isinstance(x, (int, float)) and -5.0 <= x <= 5.0:
                    range_bonus += 2
                if isinstance(y, (int, float)) and -3.0 <= y <= 3.0:
                    range_bonus += 2
                if isinstance(timeout, (int, float)) and 1.0 <= timeout <= 60.0:
                    range_bonus += 2
            
            elif op == 'aim_hand':
                x, y, z = action.get('x', 0), action.get('y', 0), action.get('z', 0)
                if isinstance(x, (int, float)) and 0.60 <= x <= 0.90:
                    range_bonus += 3
                if isinstance(y, (int, float)) and -0.5 <= y <= 0.5:
                    range_bonus += 2
                if isinstance(z, (int, float)) and -0.2 <= z <= 0.65:
                    range_bonus += 2
        
        score += min(range_bonus, 20)  # Cap bonus at 20 points
        
        # Penalty for issues (range violations, invalid ops, etc.)
        score -= len(issues) * 5
        score = max(0, score)
        
        # Debug scoring breakdown
        scoring_breakdown = {
            "step_count_score": 30 if num_generated == num_expected else (20 if abs(num_generated - num_expected) <= 1 else (10 if abs(num_generated - num_expected) <= 2 else 0)),
            "required_actions_score": (20 if key_actions["stand"] else 0) + (20 if task_name == "task1_simple" and key_actions["aim_hand"] else (20 if task_name == "task2_medium" and key_actions["move_relative"] else (20 if task_name == "task3_complex" and (key_actions["aim_hand"] and key_actions["detect_label"] and key_actions["pick_from_pixel"]) else 0))),
            "range_bonus": min(range_bonus, 20),
            "penalties": len(issues) * 5,
            "final_score": score
        }
        
        return {
            "score": score,
            "max_score": max_score,
            "num_generated": num_generated,
            "num_expected": num_expected,
            "key_actions_present": key_actions,
            "issues": issues,
            "invalid_operations": invalid_ops,
            "yaw_issues": yaw_issues,
            "scoring_breakdown": scoring_breakdown
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all prompt tests on all tasks."""
        results = {}
        
        print("=== NATURAL LANGUAGE CONTROL PROMPT TESTING ===\n")
        print(f"Testing {len(self.prompts)} prompts on {len(self.test_tasks)} tasks")
        print("=" * 60)
        
        for prompt_key, prompt_data in self.prompts.items():
            print(f"\nTesting prompt: {prompt_data['name']}")
            print(f"Description: {prompt_data['description']}")
            print("=" * 50)
            
            prompt_results = {}
            
            for task_key, task_data in self.test_tasks.items():
                print(f"\nTesting task: {task_data['complexity']}")
                print(f"Description: {task_data['description']}")
                print(f"Expected steps: {task_data['expected_steps']}")
                
                # Test the prompt
                test_result = self.test_prompt_on_task(
                    prompt_data['content'],
                    task_data['description'],
                    task_key
                )
                
                if test_result.get('success'):
                    print(f"✓ LLM generated {test_result['num_actions']} actions")
                    
                    # Show generated actions vs ground truth
                    print(f"  Generated: {json.dumps(test_result['actions'], indent=2)}")
                    print(f"  Ground Truth: {json.dumps(task_data['ground_truth'], indent=2)}")
                    
                    # Evaluate quality
                    quality = self.evaluate_plan_quality(
                        test_result['actions'],
                        task_data['ground_truth'],
                        task_key
                    )
                    
                    print(f"  Quality Score: {quality['score']}/{quality['max_score']}")
                    if 'scoring_breakdown' in quality:
                        print(f"  Scoring Breakdown:")
                        print(f"    Step count: {quality['scoring_breakdown']['step_count_score']}/30")
                        print(f"    Required actions: {quality['scoring_breakdown']['required_actions_score']}/40")
                        print(f"    Range bonus: {quality['scoring_breakdown']['range_bonus']}/20")
                        print(f"    Penalties: -{quality['scoring_breakdown']['penalties']}")
                    if quality['issues']:
                        print(f"  Issues: {', '.join(quality['issues'])}")
                    
                    prompt_results[task_key] = {
                        "test_result": test_result,
                        "quality": quality,
                        "ground_truth": task_data['ground_truth']
                    }
                else:
                    print(f"✗ LLM failed: {test_result.get('error', 'Unknown error')}")
                    if 'cleaned_response' in test_result:
                        print(f"  Cleaned response: {test_result['cleaned_response']}")
                    
                    prompt_results[task_key] = {
                        "test_result": test_result,
                        "quality": None,
                        "ground_truth": task_data['ground_truth']
                    }
                
                # Add delay to avoid rate limiting
                time.sleep(1)
            
            results[prompt_key] = prompt_results
            print("\n" + "=" * 50 + "\n")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of all test results."""
        print("\n=== TEST RESULTS SUMMARY ===\n")
        
        # Calculate overall scores for each prompt
        prompt_scores = {}
        
        # Collect all issues for summary
        all_issues = []
        
        for prompt_key, prompt_data in self.prompts.items():
            total_score = 0
            num_tasks = 0
            
            if prompt_key in results:
                for task_key, task_result in results[prompt_key].items():
                    if task_result['quality']:
                        total_score += task_result['quality']['score']
                        num_tasks += 1
                        # Collect issues
                        if task_result['quality']['issues']:
                            all_issues.extend(task_result['quality']['issues'])
                
                if num_tasks > 0:
                    avg_score = total_score / num_tasks
                    prompt_scores[prompt_key] = avg_score
        
        # Sort prompts by average score
        sorted_prompts = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Prompt Rankings (by average quality score):")
        print("-" * 50)
        for i, (prompt_key, avg_score) in enumerate(sorted_prompts, 1):
            prompt_name = self.prompts[prompt_key]['name']
            print(f"{i}. {prompt_name}: {avg_score:.1f}/100")
        
        # Show summary of issues found
        if all_issues:
            print(f"\nIssues Found ({len(all_issues)} total):")
            print("-" * 50)
            # Count and show unique issues
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {issue} (found {count} times)")
        else:
            print("\nNo issues found in any tests!")
        
        print("\nDetailed Results:")
        print("-" * 50)
        
        for prompt_key, prompt_data in self.prompts.items():
            if prompt_key in results:
                print(f"\n{prompt_data['name']}:")
                for task_key, task_result in results[prompt_key].items():
                    task_name = self.test_tasks[task_key]['complexity']
                    if task_result['quality']:
                        score = task_result['quality']['score']
                        print(f"  {task_name}: {score}/100")
                    else:
                        print(f"  {task_name}: Failed")
    
    def save_results(self, results: Dict[str, Any], filename: str = "prompt_testing/prompt_test_results.json"):
        """Save test results to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def test_single_prompt(self, prompt_key: str, task_key: str = None):
        """Test a single prompt on a single task or all tasks."""
        if prompt_key not in self.prompts:
            print(f"Prompt '{prompt_key}' not found")
            return
        
        prompt_data = self.prompts[prompt_key]
        print(f"Testing prompt: {prompt_data['name']}")
        print(f"Description: {prompt_data['description']}")
        print("=" * 50)
        
        if task_key:
            if task_key not in self.test_tasks:
                print(f"Task '{task_key}' not found")
                return
            tasks_to_test = {task_key: self.test_tasks[task_key]}
        else:
            tasks_to_test = self.test_tasks
        
        results = {}
        for tk, task_data in tasks_to_test.items():
            print(f"\nTesting task: {task_data['complexity']}")
            print(f"Description: {task_data['description']}")
            
            test_result = self.test_prompt_on_task(
                prompt_data['content'],
                task_data['description'],
                tk
            )
            
            if test_result.get('success'):
                print(f"✓ LLM generated {test_result['num_actions']} actions")
                print(f"Actions: {json.dumps(test_result['actions'], indent=2)}")
                print(f"Ground Truth: {json.dumps(task_data['ground_truth'], indent=2)}")
                
                quality = self.evaluate_plan_quality(
                    test_result['actions'],
                    task_data['ground_truth'],
                    tk
                )
                print(f"Quality Score: {quality['score']}/{quality['max_score']}")
                
                results[tk] = {"test_result": test_result, "quality": quality, "ground_truth": task_data['ground_truth']}
            else:
                print(f"✗ LLM failed: {test_result.get('error', 'Unknown error')}")
                if 'cleaned_response' in test_result:
                    print(f"  Cleaned response: {test_result['cleaned_response']}")
                
                results[tk] = {"test_result": test_result, "quality": None, "ground_truth": task_data['ground_truth']}
            
            time.sleep(1)
        
        return results

def main():
    """Main function to run the prompt testing."""
    print("Natural Language Control Prompt Testing")
    print("=" * 50)
    print("This system requires prompts.py to function.")
    print("=" * 50)
    
    # Check OpenAI API
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Show available prompts
    print("Available prompts:")
    prompt_names = get_prompt_names()
    for i, name in enumerate(prompt_names, 1):
        prompt = get_prompt(name)
        print(f"{i}. {name}: {prompt['name']} - {prompt['description']}")
    
    # Ask user which prompts to test
    print(f"\nOptions:")
    print("1. Test all prompts on all tasks")
    print("2. Test specific prompts on all tasks")
    print("3. Test a specific prompt on a specific task")
    print("4. Test a specific prompt on all tasks")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Test all prompts
        tester = PromptTester()
        results = tester.run_all_tests()
        tester.print_summary(results)
        tester.save_results(results)
        
    elif choice == "2":
        # Test specific prompts
        print("\nEnter prompt names to test (comma-separated):")
        print(f"Available: {', '.join(prompt_names)}")
        selected_prompts = input("Prompt names: ").strip().split(',')
        selected_prompts = [p.strip() for p in selected_prompts]
        
        tester = PromptTester(prompt_names_to_test=selected_prompts)
        results = tester.run_all_tests()
        tester.print_summary(results)
        tester.save_results(results)
        
    elif choice == "3":
        # Test specific prompt on specific task
        prompt_key = input("Enter prompt name: ").strip()
        task_key = input("Enter task key (task1_simple, task2_medium, task3_complex): ").strip()
        
        tester = PromptTester(prompt_names_to_test=[prompt_key])
        results = tester.test_single_prompt(prompt_key, task_key)
        
    elif choice == "4":
        # Test specific prompt on all tasks
        prompt_key = input("Enter prompt name: ").strip()
        
        tester = PromptTester(prompt_names_to_test=[prompt_key])
        results = tester.test_single_prompt(prompt_key)
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
