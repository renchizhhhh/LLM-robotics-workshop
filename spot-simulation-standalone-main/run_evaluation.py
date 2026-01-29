#!/usr/bin/env python3
"""
Wrapper script to run LLM plan evaluation from the project root.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the evaluation
from evaluate_plans import main

if __name__ == "__main__":
    # Update file paths to use evaluation_files folder
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if not arg.startswith('-') and not os.path.isabs(arg):
                # Check if it's a JSON file and add evaluation_files path
                if arg.endswith('.json'):
                    sys.argv[i] = f"evaluation_files/{arg}"
    
    main()