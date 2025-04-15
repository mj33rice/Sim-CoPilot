import os
import pandas as pd
import re
from collections import Counter

def fix_first_line_indentation(code_text, program_type="Python"):
    """
    Fixes inconsistent indentation in the first line of code.
    
    Args:
        code_text (str): The code text to fix
        program_type (str): Programming language (Python or Java)
        
    Returns:
        str: Code with normalized indentation
    """
    if not code_text or not isinstance(code_text, str):
        return code_text
        
    # Split code into lines
    lines = code_text.split('\n')
    if len(lines) <= 1:
        return code_text  # Nothing to fix if there's only one line
    
    # Check if first line has different indentation pattern than the rest
    first_line = lines[0]
    if not first_line.strip():
        return code_text  # Skip if first line is empty
    
    # Get indentation of first line
    first_indent = len(first_line) - len(first_line.lstrip())
    
    # Find common indentation in subsequent lines
    indentation_levels = []
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip())
            indentation_levels.append(indent)
    
    if not indentation_levels:
        return code_text  # No other lines with content to compare
    
    # Find the most common indentation level
    if program_type == "Python":
        # For Python, we need to be extra careful with indentation
        common_indents = Counter(indentation_levels).most_common(2)
        
        # If first line has zero indent but others are consistently indented
        if (first_indent == 0 and common_indents[0][0] > 0):
            # Check if the first line needs indentation (based on Python syntax)
            if _should_indent_python_line(first_line, lines[1:]):
                lines[0] = ' ' * common_indents[0][0] + first_line
                return '\n'.join(lines)
    else:
        # For other languages, just normalize based on most common pattern
        common_indent = Counter(indentation_levels).most_common(1)[0][0]
        if (first_indent == 0 and common_indent > 0) or (first_indent > 0 and common_indent == 0):
            if common_indent > 0:
                lines[0] = ' ' * common_indent + first_line
            else:
                lines[0] = first_line.lstrip()
            return '\n'.join(lines)
    
    return code_text  # No change needed

def _should_indent_python_line(first_line, other_lines):
    """Helper to determine if a Python line should be indented based on context"""
    # Lines that follow 'if', 'for', 'def', 'class', etc. should be indented
    indent_keywords = ['if', 'elif', 'else:', 'for', 'while', 'def', 'class', 'with', 'try:', 'except']
    
    # Check subsequent lines to see if they look like a code block
    if any(line.strip() for line in other_lines):
        # If first line looks like control flow that should have a block
        # and subsequent lines are indented, first line probably shouldn't be indented
        for keyword in indent_keywords:
            if first_line.strip().startswith(keyword) and first_line.rstrip().endswith(':'):
                return False
                
        # If first line doesn't look like the start of a block,
        # but subsequent lines are indented, first line probably should be indented too
        return True
    
    return False

def process_csv_files(directory, model_filter=None):
    """
    Process CSV files containing code generation results to fix indentation issues.
    
    Args:
        directory (str): Directory containing CSV files
        model_filter (list): Optional list of model names to filter
    """
    fixed_count = 0
    files_processed = 0
    
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        # Optionally filter by model name
        if model_filter and not any(model in filename for model in model_filter):
            continue
            
        filepath = os.path.join(directory, filename)
        files_processed += 1
        
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Check if it has the expected columns
            if 'Generated Code' not in df.columns:
                print(f"Skipping {filename}: Missing 'Generated Code' column")
                continue
                
            # Determine the programming language (if you have it in the dataset)
            program_type = "Python"  # Default to Python, adjust if you have this info
            
            # Fix indentation in the Generated Code column
            df['Fixed Generated Code'] = df['Generated Code'].apply(
                lambda code: fix_first_line_indentation(code, program_type)
            )
            
            # Count how many were fixed
            fixed = (df['Generated Code'] != df['Fixed Generated Code']).sum()
            fixed_count += fixed
            
            if fixed > 0:
                print(f"Fixed {fixed} entries in {filename}")
                
                # Save the fixed version with a new column
                new_filepath = os.path.join(directory, f"fixed_{filename}")
                df.to_csv(new_filepath, index=False)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Processed {files_processed} files, fixed {fixed_count} code snippets with indentation issues")

# Example usage
model_filter = ["claude-3-5-haiku", "claude-3-7-sonnet", "claude-3-opus"]  # Focus on Claude models
process_csv_files("./storage_server/COLM_res_update/Python/Updated_post_process", model_filter)