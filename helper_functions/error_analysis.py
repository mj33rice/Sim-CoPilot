import re
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
plt.style.use('default')
CSV_EXTENSION = '.csv'
SINGLE_FILE_KEY = 'single_file'
import warnings
warnings.filterwarnings('ignore')

def process_horizon_categories_output(row):
    # Check if the row is a string
    if isinstance(row, str):
        # Define the regular expression pattern
        pattern = r"(?P<type>\w+) '.*?' used at line (?P<usage_line>\d+) is defined at line (?P<def_line>\d+)"

        # Use the re.findall function to find all matches in the row
        matches = re.findall(pattern, row)

        # Convert the matches to the required format
        results = []
        for match in matches:
            type, usage_line, def_line = match
            results.append((type, int(usage_line), int(def_line)))

        return results
    else:
        # If the row is not a string, return an empty list
        return []

def expand_lists_to_rows(df, column):
    # Create a new DataFrame where each item in the 'Processed' column is expanded into its own row
    expanded_df = df.explode(column)

    # Split the 'Processed' column into separate 'Type', 'usage_line', and 'def_line' columns
    expanded_df[['Type', 'usage_line', 'def_line']] = pd.DataFrame(expanded_df[column].tolist(), index=expanded_df.index)
    
    # Calculate the absolute differences between 'usage_line' and 'def_line'
    expanded_df['abs_diff'] = abs(expanded_df['usage_line'] - expanded_df['def_line'])

    # Drop the 'Processed' column
    expanded_df = expanded_df.drop(columns=[column])

    return expanded_df

def process_max_range(expanded_df, groupby_col=['code_task', 'start_line', 'end_line']):
    # Drop NaN values
    expanded_df = expanded_df.dropna(subset=groupby_col)
    # Reset the index
    expanded_df = expanded_df.reset_index(drop=True)
    # Find the index of the row with the maximum difference for each group
    idx = expanded_df.groupby(groupby_col)['abs_diff'].idxmax()
    # Print out the number of rows in the original DataFrame, dropped, and the number of rows after the groupby operation
    print(f"Original DataFrame: {len(expanded_df)} rows")
    # print(f"Rows dropped: {len(expanded_df) - len(idx)}")
    print(f"DataFrame after groupby: {len(idx)} rows")
    # Keep only the rows with the maximum difference
    expanded_df = expanded_df.loc[idx]
    return expanded_df

def extract_info(csv_file):
    # Remove the file extension
    name_without_ext = csv_file.split('.csv')[0]

    # Split the name into parts
    parts = name_without_ext.split('_')

    # Extract the information
    info = {
        'model_name': parts[0],
        'gen_mode': '_'.join(parts[1:3]),
        'task': parts[3],
        'time': '_'.join(parts[4:])
    }

    return info

def group_by_res(folder_path, group_by_keys=['model_name','gen_mode'], group_gpt_4_turbo=True):
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Create a dictionary to store the groups
    groups = defaultdict(list)

    # Group the CSV files
    for csv_file in csv_files:
        # Extract the information from the file name
        info = extract_info(csv_file)

        # Get the group key
        group_key_parts = [info[key] for key in group_by_keys]
        group_key = '_'.join(group_key_parts)

        # If group_gpt_4_turbo is True, treat 'gpt-4-turbo' and 'gpt-4-turbo-2024-04-09' as the same model_name
        if group_gpt_4_turbo and 'model_name' in group_by_keys and 'gpt-4-turbo' in group_key:
            group_key = group_key.replace('gpt-4-turbo-2024-04-09', 'gpt-4-turbo')

        # Add the CSV file to the group
        groups[group_key].append(csv_file)

    # Concatenate the CSV files in each group
    for group_key, group_files in groups.items():
        print(f"Grouped by {group_by_keys} and combined {', '.join(group_files)}")
        dfs = [pd.read_csv(os.path.join(folder_path, csv_file)) for csv_file in group_files]
        df_group = pd.concat(dfs, ignore_index=True)
        groups[group_key] = df_group

    return groups

def groupby_sanity_check(grouped_dfs):
    for group_key, df in grouped_dfs.items():
        print(f"Group Key: {group_key}")
        print(len(df))
        print("\n")
        
def process_folder(folder_path: str) -> Dict[str, pd.DataFrame]:
    global col_to_check
    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Group and concatenate the CSV files
        grouped_dfs = group_by_res(folder_path)
    elif os.path.isfile(folder_path) and folder_path.endswith(CSV_EXTENSION):
        # If the path is a CSV file, read it into a DataFrame
        grouped_dfs = {SINGLE_FILE_KEY: pd.read_csv(folder_path)}
    else:
        # If the path is not a directory and not a CSV file, raise an error
        raise ValueError(f"Invalid path: {folder_path}. Path must be a directory containing CSV files or a CSV file.")
    processed_dfs = {}
    for group_key, df in grouped_dfs.items():
        init_row_num = len(df)
        # Columns to check for NaN values
        initial_cols_to_check = ['horizon_categories_output', 'horizon_freq_analysis']

        # Drop rows with NaN values in the initial_cols_to_check
        df_after_initial_drop = df.dropna(subset=initial_cols_to_check)

        # Process the 'horizon_categories_output' column
        df_after_initial_drop['Processed'] = df_after_initial_drop['horizon_categories_output'].apply(process_horizon_categories_output)

        # Expand the lists to rows
        expanded_df = expand_lists_to_rows(df_after_initial_drop, 'Processed')

        # Additional columns to check for NaN values
        additional_cols_to_check = ['horizon_categories_output', 'horizon_freq_analysis', 'Type', 'usage_line', 'def_line', 'abs_diff', col_to_check]

        # Drop rows with NaN values in the additional_cols_to_check
        expanded_df_filtered = expanded_df.dropna(subset=additional_cols_to_check)

        print("#############################################")
        # Print out the number of rows in the original DataFrame, and dropped due to NaN values in the cols_to_check
        print(f"Original DataFrame: {init_row_num} rows")
        print(f"Rows dropped due to NaN in initial columns {initial_cols_to_check}: {init_row_num - len(df_after_initial_drop)}")
        print(f"Rows after expansion: {len(expanded_df)} rows")
        print(f"Rows dropped due to NaN in additional columns {additional_cols_to_check}: {len(expanded_df) - len(expanded_df_filtered)}")

        # Process the max range
        df = process_max_range(expanded_df_filtered)

        processed_dfs[group_key] = df

    return processed_dfs

def convert_to_float(val: str) -> float:
    try:
        return float(val)
    except ValueError:
        parts = val.split('/')
        if len(parts) == 2:
            num, denom = parts
            if denom != '0':
                return float(num) / float(denom)
        return 0.0
    
def draw_histogram(df, group_key, show_pass_dist=False, bins=30):
    global col_to_check
    plt.figure(dpi=400)
    # Calculate the absolute differences between 'usage_line' and 'def_line'
    differences = df['abs_diff']

    # Define the bin edges
    bin_edges = np.linspace(differences.min(), differences.max(), bins+1)

    # Create a histogram of the differences
    plt.hist(differences, bins=bin_edges, edgecolor='black', alpha=0.5, label='All data')

    if show_pass_dist:
        # Convert 'col_to_check' to float and filter the DataFrame where it is 1
        df[col_to_check] = df[col_to_check].apply(convert_to_float)
        filtered_df = df[df[col_to_check] == 1]

        # Create a histogram of the differences for the filtered DataFrame
        plt.hist(filtered_df['abs_diff'], bins=bin_edges, edgecolor='black', color='red', alpha=0.5, label='gen_code_passed')

    # Set the title and labels
    plt.title(group_key)
    plt.xlabel('Recall Distance')
    plt.ylabel('Frequency')

    # Add a legend
    plt.legend()

    # Display the histogram
    plt.show()

def plot_histograms(groups, show_pass_dist=True):
    global col_to_check
    for group_key, df_group in groups.items():
        print(f"Plotting histogram for group: {group_key}")
        draw_histogram(df_group, group_key, show_pass_dist)

def classify_and_calculate(df, short_range, medium_range, report_err_bar=True):
    global col_to_check
    def classify_abs_diff(x):
        if x <= short_range:
            return 'Short'
        elif x <= medium_range:
            return 'Medium'
        else:
            return 'Long'

    df['range_class'] = df['abs_diff'].apply(classify_abs_diff)
    df[col_to_check] = df[col_to_check].apply(convert_to_float)

    total_counts = df.groupby('range_class').size()
    pass_counts = df[df[col_to_check] == 1].groupby('range_class').size()
    # Reindex the pass_counts to include all range classes & fill 0 for NaN values
    pass_counts = pass_counts.reindex(total_counts.index, fill_value=0)
    if report_err_bar:
        percentages, err_bar = cal_err_bar(pass_counts, total_counts)
        err_bar = err_bar.values
    else:
        percentages = pass_counts / total_counts * 100
        err_bar = None

    result_df = pd.DataFrame({
        'range_class': total_counts.index,
        'total_counts': total_counts.values,
        'passed_counts': pass_counts.values,
        'percentages': percentages.values
    })

    result_df['range_class'] = pd.Categorical(result_df['range_class'], categories=['Short', 'Medium', 'Long'], ordered=True)
    result_df = result_df.sort_values('range_class')
    # result_df['percentages'] = result_df['percentages'].apply(lambda x: '{0:.3f}%'.format(x))
    if report_err_bar:
        result_df['percentages'] = result_df.apply(lambda row: f"{row['percentages']*100:.1f} ± {err_bar[row.name]*100:.1f}", axis=1)
    else:
        result_df['percentages'] = result_df['percentages'].apply(lambda x: f"{x:.1f}%")

    return percentages, result_df

# percentages, result_df = classify_and_calculate(combined_df['gpt-4-turbo_with_afterlines'], 30, 100)
# result_df

def process_all_dfs(combined_df, short_range, medium_range):
    all_results = []
    for key in combined_df:
        percentages, result_df = classify_and_calculate(combined_df[key], short_range, medium_range)
        result_df.insert(0, 'key', key)
        all_results.append(result_df)
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df

def split_df(final_df):
    unique_keys = final_df['key'].unique()
    sub_dfs = {key: final_df[final_df['key'] == key].drop(columns='key') for key in unique_keys}
    return sub_dfs

def process_and_plot(folder_path, col_to_check):
    # Set the global variable
    combined_df = process_folder(folder_path)
    # Call the function
    # plot_histograms(combined_df)
    return combined_df

def process_and_display(combined_df, short_range, medium_range):
    # Set the short and medium ranges
    final_df = process_all_dfs(combined_df, short_range, medium_range)

    sub_dfs = split_df(final_df)
    # Sort the keys based on model_name and gen_mode
    sorted_keys = sorted(sub_dfs.keys(), key=lambda x: (x.rsplit('_', 1)[0], x.rsplit('_', 1)[1]))

    for key in sorted_keys:
        df = sub_dfs[key]
        print(f"Key: {key}")
        display(df)
    return sub_dfs

def bootstrap_resampling(pass_count, total_count, num_resamples=10000):
    # Calculate model's performance
    performance = pass_count / total_count

    # Generate bootstrap resamples
    resamples = np.random.choice([0, 1], size=(num_resamples, total_count), p=[1-performance, performance])

    # Calculate pass count for each resample
    resample_pass_counts = resamples.sum(axis=1)

    # Calculate performance for each resample
    resample_performances = resample_pass_counts / total_count

    # Calculate average and 1.96 standard deviations of resample performances
    avg_performance = resample_performances.mean()
    std_dev_performance = resample_performances.std()

    return avg_performance, 1.96 * std_dev_performance

def cal_err_bar(pass_counts, total_counts, num_resamples=10000):
    percentages = []
    err_bars = []
    for pass_count, total_count in zip(pass_counts, total_counts):
        # Use bootstrap resampling to calculate average performance and error bar
        percentage, err_bar = bootstrap_resampling(pass_count, total_count, num_resamples)
        percentages.append(percentage)
        err_bars.append(err_bar)

    return pd.Series(percentages, index=total_counts.index), pd.Series(err_bars, index=total_counts.index)

def save_df_dict_to_csv_with_keys(df_dict, output_file_name):
    final_df = pd.DataFrame()

    for key, df in df_dict.items():
        key_row = pd.DataFrame({col: [key] if col == list(df.columns)[0] else [pd.NA] for col in df.columns})
        final_df = pd.concat([final_df, key_row, df], ignore_index=True)

    final_df.to_csv(output_file_name, index=False)

def refine_error_msg(df):
    # Initialize an empty list to dynamically collect error types
    dynamic_errors = []

    def extract_error_type(post_eval_res, current_category):
        nonlocal dynamic_errors
        # Only refine if current category starts with "Error"
        if not current_category.startswith("Error"):
            return current_category
        
        # Handle case where 'post_process_eval_res' is a list
        if isinstance(post_eval_res, list):
            if all("success" in res.lower() for res in post_eval_res):
                return "Success"
            for failure_indicator in ["Failure: ", "Timeout: "]:
                if any(failure_indicator in res for res in post_eval_res):
                    return failure_indicator.strip()
        
        # Clean the post_eval_res from unwanted characters and split by error types
        cleaned_post_eval_res = post_eval_res.replace('\n', '').replace('n', '')
        found_errors = re.findall(r'\b\w*Error\b', cleaned_post_eval_res)
        
        if found_errors:
            for error in found_errors:
                if error not in dynamic_errors:
                    dynamic_errors.append(error)

            # Iterate through found_errors, prioritizing non-"Error" types
            refined_error = "Error"  # Default to "Error" if no other types are found
            for error in reversed(found_errors):  # Reverse to prioritize the last found error
                if error != "Error":
                    refined_error = error
                    break  # Stop at the first non-"Error" type found from the end
        else:
            # If no specific error is found, try to identify other types of issues
            if "Timeout" in cleaned_post_eval_res:
                refined_error = "Timeout"
            elif "Failure" in cleaned_post_eval_res:
                refined_error = "Failure"
            else:
                # If no dynamic error or other issue is found, keep the original category
                return current_category
        
        # If the refined error category is "Error" but the current category is more specific, keep the original
        if refined_error == "Error" and current_category != "Error":
            return current_category
        else:
            return refined_error

    # Apply the refinement function to each row
    df['error_category'] = df.apply(lambda row: extract_error_type(row['post_process_eval_res'], row['error_category']), axis=1)
    return df

def reorganize_error_category(df):
    # Step 1: Merge "Error ']" and "['Failure" into "Output Mismatch"
    df['error_category'] = df['error_category'].replace(["Error ']", "['Failure"], "Output Mismatch")

    # Step 3: Remove "['" from the beginning of each error_category
    df['error_category'] = df['error_category'].str.replace("^\[\s*'", "", regex=True)

    # Rename "Error Compilation" to "CompilationError"
    df['error_category'] = df['error_category'].replace("Error Compilation", "CompilationError")

    # Calculate the frequency before merging categories under 40 to "Others"
    frequency = df['error_category'].value_counts()

    # Step 2: Merge categories with frequency under 40 into "Others"
    categories_to_merge = frequency[frequency < 40].index
    df['error_category'] = df['error_category'].apply(lambda x: "Others" if x in categories_to_merge else x)

    # Recalculate frequency after all modifications
    frequency = df['error_category'].value_counts()

    print(frequency)
    return df, frequency

def calculate_frequency_from_csv(csv_file_name):
    # Step 1: Read the CSV file
    df = pd.read_csv(csv_file_name)

    # Step 2: Refine extraction of 'error_category'
    def extract_error_category(x):
        if pd.notna(x):
            parts = x.split(':')
            if parts[0] in ("['Error", '["Error'):
                # Check if there is a next word before the next ":"
                if len(parts) > 1 and parts[1].strip():
                    return 'Error ' + parts[1].split()[0]
                else:
                    return 'Error Empty'
            else:
                return parts[0]
        return x
    
    df['error_category'] = df['post_process_eval_res'].apply(extract_error_category)
    frequency = df['error_category'].value_counts()
    # Step 3: Further refine 'error_category' by looking for common error types
    df = refine_error_msg(df)
    # Step 4: Calculate the frequency of each category
    frequency = df['error_category'].value_counts()    
    return frequency, df

def find_failure_rows(df, pattern):
    # Find all rows where 'error_category' is 'Failure'
    failure_rows = df[df['error_category'] == pattern]
    return failure_rows

def sanity_check(df, frequency):
    total_rows = df.shape[0]
    total_frequency = frequency.sum()  # Summing directly since frequency is a pandas.Series

    if total_rows == total_frequency:
        print("Sanity check passed: The sum of frequencies matches the number of rows in the dataframe.")
    else:
        print(f"Sanity check failed: The sum of frequencies ({total_frequency}) does not match the number of rows in the dataframe ({total_rows}).")# Example usage


# def calculate_error_category_percentage(df, groupby_keys, target_col):
#     # Group by the specified keys, then count each target column within the groups
#     error_counts = df.groupby(groupby_keys + [target_col]).size().reset_index(name='count')

#     # Calculate the total counts for each group specified by the groupby keys
#     total_counts = df.groupby(groupby_keys).size().reset_index(name='total_count')

#     import pdb; pdb.set_trace()

#     # Merge the counts with the total counts to calculate percentages
#     error_percentage = pd.merge(error_counts, total_counts, on=groupby_keys)

#     # Calculate the percentage of each target column within its group
#     error_percentage['error_percentage'] = (error_percentage['count'] / error_percentage['total_count']) * 100

#     # Sort the results for better readability
#     error_percentage = error_percentage.sort_values(by=groupby_keys + ['error_percentage'], ascending=False)

#     return error_percentage

def calculate_error_category_percentage(df, groupby_keys, target_col, report_err_bar=True):
    # Group by the specified keys, then count each target column within the groups
    error_counts = df.groupby(groupby_keys + [target_col]).size().reset_index(name='count')

    # Calculate the total counts for each group specified by the groupby keys
    total_counts = df.groupby(groupby_keys).size().reset_index(name='total_count')


    # Merge the counts with the total counts to calculate percentages
    error_percentage = pd.merge(error_counts, total_counts, on=groupby_keys)

    # # Calculate the percentage of each target column within its group
    # error_percentage['error_percentage'] = (error_percentage['count'] / error_percentage['total_count']) * 100

    # # Sort the results for better readability
    # error_percentage = error_percentage.sort_values(by=groupby_keys + ['error_percentage'], ascending=False)
    import pdb; pdb.set_trace()
    # return error_percentage
    if report_err_bar:
        # Assuming cal_err_bar is a function that calculates the error bar and returns percentages and error bars
        percentages, err_bar = cal_err_bar(error_percentage['count'], error_percentage['total_count'])
        error_percentage['error_percentage'] = percentages * 100  # Convert to percentage
        error_percentage['error_bar'] = err_bar * 100  # Convert to percentage
        # Format the 'error_percentage' column to include error bars
        error_percentage['error_percentage'] = error_percentage.apply(lambda row: f"{row['error_percentage']:.1f} ± {row['error_bar']:.1f}", axis=1)
    else:
        # Calculate the percentage of each target column within its group
        error_percentage['error_percentage'] = (error_percentage['count'] / error_percentage['total_count']) * 100
        # Format the 'error_percentage' column as a percentage string
        error_percentage['error_percentage'] = error_percentage['error_percentage'].apply(lambda x: f"{x:.1f}%")

    # Sort the results for better readability
    error_percentage = error_percentage.sort_values(by=groupby_keys + ['error_percentage'], ascending=False)

    return error_percentage

def get_top_n_error_categories_by_count_full(error_percentage_df, groupby_keys, n):
    # Sort the DataFrame by groupby_keys and 'count' in descending order to ensure the highest counts are first
    error_percentage_df_sorted = error_percentage_df.sort_values(by=groupby_keys + ['count'], ascending=[True] * len(groupby_keys) + [False])

    # Use groupby and head to select the top n rows for each group based on the 'count'
    top_n_error_categories_full = error_percentage_df_sorted.groupby(groupby_keys).head(n).reset_index(drop=True)

    return top_n_error_categories_full

# csv_file_name = './Analysis_Results/storage_server/All_models_res/final_results_all_models_processed.csv'  # Replace with your CSV file name
csv_file_name = './storage_server/COLM_res_update/All_Models_res/final_results_all_models_processed.csv'

frequency, df = calculate_frequency_from_csv(csv_file_name)
import pdb; pdb.set_trace()

df, frequency = reorganize_error_category(df)
import pdb; pdb.set_trace()
output_csv_file_name = './storage_server/COLM_res_update/All_Models_res/final_results_all_models_error_analysis.csv'  # Replace with your CSV file name
df.to_csv(output_csv_file_name, index=False)

sanity_check(df, frequency)
import pdb; pdb.set_trace()
failure_rows = find_failure_rows(df, "Error Compilation")
print(failure_rows)

############################################
output_csv_file_name = './storage_server/COLM_res_update/All_Models_res/error_analysis/final_results_all_models_error_analysis.csv'  # Replace with your CSV file name
df = pd.read_csv(output_csv_file_name)
import pdb; pdb.set_trace()

error_percentage_df = calculate_error_category_percentage(df, ['task_type', 'model_name'], 'error_category')
print(error_percentage_df)
all_err_save_path = './storage_server/COLM_res_update/All_Models_res/error_analysis/final_results_all_models_all_err.csv'  # Replace with your CSV file name
error_percentage_df.to_csv(all_err_save_path, index=False)
import pdb; pdb.set_trace()

top_n_error_categories = get_top_n_error_categories_by_count_full(error_percentage_df, ['task_type', 'model_name'], 5)
import pdb; pdb.set_trace()

top_n_err_save_path = './storage_server/COLM_res_update/All_Models_res/error_analysis/final_results_all_models_top_n_err.csv'  # Replace with your CSV file name
top_n_error_categories.to_csv(top_n_err_save_path, index=False)

############################################
output_csv_file_name = './storage_server/COLM_res_update/All_Models_res/error_analysis/final_results_all_models_error_analysis.csv'  
#Replace with your CSV file name
df = pd.read_csv(output_csv_file_name)
import pdb; pdb.set_trace()

error_percentage_df = calculate_error_category_percentage(df, ['model_name'], 'error_category')
print(error_percentage_df)
import pdb; pdb.set_trace()
all_err_save_path = './storage_server/COLM_res_update/All_Models_res/error_analysis/final_results_all_models_all_err_by_model.csv'  # Replace with your CSV file name
error_percentage_df.to_csv(all_err_save_path, index=False)
import pdb; pdb.set_trace()

top_n_error_categories = get_top_n_error_categories_by_count_full(error_percentage_df, ['model_name'], 5)
import pdb; pdb.set_trace()

top_n_err_save_path = './storage_server/COLM_res_update/All_Models_res/error_analysis/final_results_all_models_top_n_err_by_model.csv'  # Replace with your CSV file name
top_n_error_categories.to_csv(top_n_err_save_path, index=False)

