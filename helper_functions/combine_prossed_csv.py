import os
import pandas as pd
import glob
import json
import re
import argparse

def combine_csv_files(dirs=None, root_path="", extract_n=None):
    """
    Combine CSV files from specified directories with optional file count limitation.
    
    Args:
        dirs (list, optional): List of directories to process. If None, uses default directories.
        root_path (str, optional): Root path to prepend to all directories.
        extract_n (int, optional): If provided, limits processing to this many files per directory.
    """
    # Try to load tasks_path.json to get task names
    try:
        with open('./example_code/tasks_path.json', 'r') as f:
            tasks_path = json.load(f)
        python_tasks = list(tasks_path.get("Python", {}).keys())
        java_tasks = list(tasks_path.get("Java", {}).keys())
        print(f"Found {len(python_tasks)} Python tasks and {len(java_tasks)} Java tasks")
    except Exception as e:
        print(f"Warning: Could not load tasks_path.json: {e}")
        import pdb;pdb.set_trace()
        python_tasks = []
        java_tasks = []

    # Function to extract metadata from filename
    def extract_metadata(filename, folder_path):
        base_filename = os.path.basename(filename).replace('.csv', '')
        
        # Extract model_name (part before "_with_afterlines" or "_no_afterlines")
        if "_with_afterlines" in base_filename:
            model_name = base_filename.split("_with_afterlines")[0]
            afterlines_type = "_with_afterlines"
        elif "_no_afterlines" in base_filename:
            model_name = base_filename.split("_no_afterlines")[0]
            afterlines_type = "_no_afterlines"
        else:
            model_name = "unknown"
            afterlines_type = ""
        
        # Extract code_task from the filename
        code_task = "unknown"
        
        if afterlines_type:
            parts = base_filename.split(afterlines_type + "_")
            if len(parts) > 1:
                rest = parts[1]
                
                # First try to find a direct match with known tasks
                for task in python_tasks + java_tasks:
                    if rest.startswith(task + "_") or rest == task:
                        code_task = task
                        break
                
                # If no match found, try to extract based on pattern
                if code_task == "unknown":
                    # Look for date pattern like _03_25_09_48
                    date_pattern = re.compile(r'_\d{2}_\d{2}_\d{2}_\d{2}')
                    date_match = date_pattern.search(rest)
                    if date_match:
                        code_task = rest[:date_match.start()]
                    else:
                        code_task = rest.split('_')[0]
        
        # Determine language and task type
        if "Java_all_res" in folder_path:
            lang = "java"
        elif "Python_all_res" in folder_path or "Pyhton_all_res" in folder_path:  # Handle typo
            lang = "python"
        else:
            lang = "unknown"
        
        if "Completion" in folder_path:
            mode = "completion"
        elif "Infilling" in folder_path:
            mode = "infilling"
        else:
            mode = "unknown"
        
        # Combine mode and language for task_type
        task_type = f"{mode}_{lang}"
        
        return {
            "model_name": model_name,
            "code_task": code_task,
            "task_type": task_type
        }

    # Define the directories to search in if not provided
    if dirs is None:
        dirs = [
            "All_Models_res/Java_all_res/Completion/Updated_post_process",
            "All_Models_res/Java_all_res/Infilling/Updated_post_process",
            "All_Models_res/Python_all_res/Completion/Updated_post_process",
            "All_Models_res/Python_all_res/Infilling/Updated_post_process"
        ]
    
    # Add root path to directories if provided
    if root_path:
        dirs = [os.path.join(root_path, d) for d in dirs]

    # Create an empty list to store all DataFrames
    all_dfs = []
    processed_count = 0
    sample_mode = extract_n is not None

    # Process each directory
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist. Skipping.")
            continue
        
        # Find all CSV files in the directory (not in subdirectories)
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        print(f"Found {len(csv_files)} CSV files in {dir_path}")
        
        # If in sample mode, limit the number of files processed
        if sample_mode:
            if extract_n <= 0:
                print(f"Skipping directory due to extract_n={extract_n}")
                continue
                
            csv_files = csv_files[:extract_n]
            print(f"SAMPLE MODE: Processing only {len(csv_files)} files from this directory")
        
        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Extract metadata from filename
                metadata = extract_metadata(csv_file, dir_path)
                
                # Add metadata columns to the DataFrame
                for key, value in metadata.items():
                    df[key] = value
                
                # Add source file path for reference
                df['source_file'] = csv_file
                
                # Append to the list
                all_dfs.append(df)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} files...")
                    
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save the combined DataFrame to a new CSV file
        output_suffix = f"_sample{extract_n}" if sample_mode else ""
        output_file = f"combined_results{output_suffix}.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined {len(all_dfs)} CSV files into {output_file}")
        print(f"Combined DataFrame has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")
        return combined_df
    else:
        print("No CSV files were processed successfully.")
        return None

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Combine CSV files with optional sample limit')
    parser.add_argument('--extract_n', type=int, default=None, 
                        help='Number of files to extract from each directory (for testing)')
    parser.add_argument('--root_path', type=str, default='./storage_server/COLM_res_update/',
                        help='Root path to prepend to all directories')
    parser.add_argument('--dirs', nargs='*', default=None,
                        help='List of directories to process (if not provided, uses defaults)')
    args = parser.parse_args()
    
    # Run the combination function with the specified parameters
    combine_csv_files(dirs=args.dirs, root_path=args.root_path, extract_n=args.extract_n)