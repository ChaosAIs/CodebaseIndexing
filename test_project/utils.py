"""
Utility functions for the test project.
"""

import json
from typing import List, Dict, Any

def read_config(file_path: str) -> Dict[str, Any]:
    """Read configuration from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {file_path} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in {file_path}")
        return {}

def write_config(file_path: str, config: Dict[str, Any]) -> bool:
    """Write configuration to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error writing config: {e}")
        return False

def format_list(items: List[str], separator: str = ", ") -> str:
    """Format a list of items as a string."""
    return separator.join(items)

def validate_email(email: str) -> bool:
    """Simple email validation."""
    return "@" in email and "." in email.split("@")[1]

class Logger:
    """Simple logging utility."""
    
    def __init__(self, name: str):
        self.name = name
        self.logs = []
    
    def info(self, message: str):
        """Log an info message."""
        log_entry = f"[INFO] {self.name}: {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def error(self, message: str):
        """Log an error message."""
        log_entry = f"[ERROR] {self.name}: {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.logs.copy()
