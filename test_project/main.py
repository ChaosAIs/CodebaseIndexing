"""
A simple test Python file for demonstrating the codebase indexing system.
"""

def hello_world():
    """Print a greeting message."""
    print("Hello, World!")
    return "Hello, World!"

def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    result = a + b
    print(f"The sum of {a} and {b} is {result}")
    return result

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history

if __name__ == "__main__":
    hello_world()
    print(calculate_sum(5, 3))
    
    calc = Calculator()
    print(calc.add(10, 20))
    print(calc.multiply(4, 5))
    print("History:", calc.get_history())
