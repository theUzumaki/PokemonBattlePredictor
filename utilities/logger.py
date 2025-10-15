"""
Logger module for colored and formatted console output.
"""

# ANSI color codes
COLORS = {
    'reset': '\033[0m',
    'black': '\033[30m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'gray': '\033[90m',
    'light_red': '\033[91m',
    'light_green': '\033[92m',
    'light_yellow': '\033[93m',
    'light_blue': '\033[94m',
    'light_magenta': '\033[95m',
    'light_cyan': '\033[96m',
}


def log(*text, color='reset', tabs=0, end='\n', sep=' '):
    """
    Print formatted text with color and indentation.
    
    Args:
        *text: Variable number of arguments to print
        color: Color name from COLORS dictionary (default: 'reset')
        tabs: Number of tab indentations (default: 0)
        end: String appended after the last value (default: '\\n')
        sep: String inserted between values (default: ' ')
    
    Example:
        log("Hello", "World", color='green', tabs=1)
        log("Error occurred", color='red')
        log("Item 1", "Item 2", color='cyan', tabs=2, end=' ')
    """
    # Get the color code, default to reset if color not found
    color_code = COLORS.get(color.lower(), COLORS['reset'])
    reset_code = COLORS['reset']
    
    # Create indentation
    indentation = '\t' * tabs
    
    # Join all text arguments with separator
    message = sep.join(str(item) for item in text)
    
    # Print with color and indentation
    print(f"{indentation}{color_code}{message}{reset_code}", end=end)
