import pandas as pd
import numpy as np
import os
import argparse

def calculate_average_differences_from_baseline(file_path):
    """
    This function takes the file path for a CSV file containing baseline and other metric values.
    It calculates the absolute average difference from the baseline for each metric across the dataset,
    ensuring that the differences are positive.
    
    Parameters:
    - file_path: str, the path to the CSV file.
    
    Returns:
    - average_difference: pandas Series, the absolute average differences from baseline for each metric.
    """
    # Load the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract the baseline values
    baseline_values = data.iloc[0, 2:]
    
    # Directly subtract the baseline values from the corresponding columns for the rest of the data
    difference = data.iloc[1:, 2:].subtract(baseline_values, axis=1)
    
    # Take the absolute values to ensure the differences are positive
    absolute_difference = difference.abs()
    
    # Calculate the average difference for each metric
    average_difference = absolute_difference.mean()
    
    return average_difference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--csv_in_path', type=str, default='/hdd/wi/sscd-copy-detection/score_pre_metrics(importance).csv', help='Path to input CSV file')
    args = parser.parse_args()
    
    # Calculate the average differences from baseline for each metric
    average_difference = calculate_average_differences_from_baseline(args.csv_in_path)
    
    # Print the results
    print(average_difference)
    
    # Save the results to a CSV file
    # average_difference.to_csv('average_difference.csv', index=False)