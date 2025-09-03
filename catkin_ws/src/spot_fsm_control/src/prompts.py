#!/usr/bin/env python

"""
Prompts file for natural language control testing.
Add new prompts here and reference them by name in the testing system.
"""

# Dictionary of all available prompts
PROMPTS = {
    "minimal": {
        "name": "Minimal Prompt",
        "description": "Basic prompt with just available actions",
                        "content": """
        
Available actions:
- {"op":"stand","height":float}
- {"op":"sit"}
- {"op":"move_relative","x":float,"y":float,"yaw":float,"timeout":float}
- {"op":"aim_hand","x":float,"y":float,"z":float}
- {"op":"detect_label","label":str,"conf":float,"image_source":"hand_color_image"}
- {"op":"pick_from_pixel","image_source":"hand_color_image","x":0,"y":0,"top_down":bool,"auto_walk":bool,"timeout":float}

IMPORTANT: Return ONLY a valid JSON array of actions. Ensure all quotes, colons, and commas are properly formatted."""
    },
    "constrained": {
        "name": "Constrained Prompt",
        "description": "Prompt with key constraints and limitations",
        "content": """

IMPORTANT CONSTRAINTS:
- Yaw values must be in RADIANS (not degrees)
- Robot must STAND before performing any other actions
- Movement limits (move_relative):
  - x: [-5.0, 5.0] meters (forward/backward)
  - y: [-3.0, 3.0] meters (left/right)
  - timeout: [1.0, 60.0] seconds
- Standing limits:
  - height: [-0.2, 0.5] meters
- Hand camera limits (aim_hand):
  - x: [0.60, 0.90] meters forward from BODY
  - y: [-0.5, 0.5] meters left/right
  - z: [-0.2, 0.65] meters height
- Don't do any more actions than necessary for fulfilling the task
- Confidece must be less than or equal to 0.01

Available actions:
- {"op":"stand","height":float}
- {"op":"sit"}
- {"op":"move_relative","x":float,"y":float,"yaw":float,"timeout":float}
- {"op":"aim_hand","x":float,"y":float,"z":float}
- {"op":"detect_label","label":str,"conf":float,"image_source":"hand_color_image"}
- {"op":"pick_from_pixel","image_source":"hand_color_image","x":0,"y":0,"top_down":bool,"auto_walk":bool,"timeout":float}

IMPORTANT: Return ONLY a valid JSON array of actions. Ensure all quotes, colons, and commas are properly formatted."""
    }
}

def get_prompt(prompt_name: str):
    """Get a specific prompt by name."""
    if prompt_name in PROMPTS:
        return PROMPTS[prompt_name]
    else:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(PROMPTS.keys())}")

def get_all_prompts():
    """Get all available prompts."""
    return PROMPTS.copy()

def get_prompt_names():
    """Get list of all prompt names."""
    return list(PROMPTS.keys())

def add_prompt(name: str, display_name: str, description: str, content: str):
    """Add a new prompt to the collection."""
    PROMPTS[name] = {
        "name": display_name,
        "description": description,
        "content": content
    }
    print(f"Added new prompt: {display_name}")

def remove_prompt(name: str):
    """Remove a prompt from the collection."""
    if name in PROMPTS:
        removed = PROMPTS.pop(name)
        print(f"Removed prompt: {removed['name']}")
        return True
    else:
        print(f"Prompt '{name}' not found")
        return False

def list_prompts():
    """List all available prompts with descriptions."""
    print("Available Prompts:")
    print("=" * 50)
    for name, prompt in PROMPTS.items():
        print(f"{name}: {prompt['name']}")
        print(f"  Description: {prompt['description']}")
        print(f"  Content preview: {prompt['content'][:100]}...")
        print("-" * 30)

if __name__ == "__main__":
    # Show available prompts when run directly
    list_prompts()
