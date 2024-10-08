import os
import warnings
from bqtmizer.bootqa import random_sample_unique
from bqtmizer.bootqa import bootstrap_sampling, run_qpu, merge, print_results
import pandas as pd


def validate_and_resolve_path(user_input: str) -> str:
    """
    Validate a file path and convert it to an absolute path.

    Args:
        user_input (str): The file path to validate (can be absolute or relative).

    Returns:
        str: The validated absolute path.

    Raises:
        ValueError: If the path contains invalid characters or is not a CSV file.
        FileNotFoundError: If the path does not exist.
    """
    # # Get the current OS path separator
    # separator = os.path.sep
    #
    # # Define a regex pattern for valid path characters
    # pattern = r'^[a-zA-Z0-9_ :/\\]+$' if separator == '/' else r'^[a-zA-Z0-9_ :\\]+$'
    #
    # # Check for the correct separator and any invalid characters
    # if not re.match(pattern, user_input):
    #     raise ValueError(f"Invalid path: {user_input}. Use '{separator}' as the path separator.")

    # Convert relative paths to absolute paths
    absolute_path = os.path.abspath(user_input)

    # Check if the resolved path exists
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"File not found: {absolute_path}")

    # Check if the file is a CSV
    if not is_csv_file(absolute_path):
        raise ValueError(f"Invalid file type: {absolute_path}. Please provide a CSV file.")

    return absolute_path


def is_csv_file(file_path: str) -> bool:
    """
    Check if the given file is a CSV file based on its extension.

    Args:
        file_path (str): The file path to check.

    Returns:
        bool: True if the file is a CSV, False otherwise.
    """
    return file_path.lower().endswith('.csv')


def validate_weights(weights: dict, effectiveness_list: list, cost_list: list) -> None:
    """
    Validate the weights dictionary according to the following rules:
    1. The sum of the values should be 1.
    2. The dictionary must contain a "num" element.
    3. All other elements must match the ones in effectiveness_list and cost_list.

    Args:
        weights (dict): The weights dictionary to validate.
        effectiveness_list (list): The list of effectiveness metrics.
        cost_list (list): The list of cost metrics.

    Raises:
        ValueError: If any validation rule is violated.
    """
    # Ensure that the sum of weights is equal to 1
    total_weight = sum(weights.values())
    if not abs(total_weight - 1) < 1e-6:  # Allow for small floating-point precision errors
        raise ValueError(f"The sum of the weights should be 1. Current sum is {total_weight}.")

    # Ensure that the "num" element is present
    if "num" not in weights:
        raise ValueError('The weights dictionary must contain the "num" element.')

    # Check that all keys in the weights (except "num") are in effectiveness_list or cost_list
    expected_keys = set(effectiveness_list + cost_list)
    provided_keys = set(weights.keys()) - {"num"}

    # Check for missing elements in weights
    missing_keys = expected_keys - provided_keys
    if missing_keys:
        raise ValueError(f"The weights dictionary is missing keys: {missing_keys}")

    # Check for extra elements not in effectiveness_list or cost_list
    extra_keys = provided_keys - expected_keys
    if extra_keys:
        raise ValueError(f"The weights dictionary contains extra keys: {extra_keys}")


def validate_lists(effectiveness_list: list, cost_list: list, column_names: set) -> None:
    """
    Validate effectiveness_list and cost_list according to the following rules:
    1. There should be no overlapping elements between the two lists.
    2. Neither list should contain an element called "num".
    3. All elements in both lists should be valid column names in the CSV file.

    Args:
        effectiveness_list (list): List of effectiveness metrics.
        cost_list (list): List of cost metrics.
        column_names (set): Set of column names from the CSV file.

    Raises:
        ValueError: If any validation rule is violated.
    """
    # Ensure there are no overlapping elements between the two lists
    overlap = set(effectiveness_list) & set(cost_list)
    if overlap:
        raise ValueError(f"Effectiveness and cost lists cannot have overlapping elements: {overlap}")

    # Ensure neither list contains the element "num"
    if "num" in effectiveness_list or "num" in cost_list:
        raise ValueError('The element "num" is not allowed in the effectiveness or cost list.')

    # Ensure all elements are valid column names from the dataset
    invalid_effectiveness = set(effectiveness_list) - column_names
    invalid_cost = set(cost_list) - column_names

    if invalid_effectiveness:
        raise ValueError(f"The following effectiveness elements are not valid column names: {invalid_effectiveness}")

    if invalid_cost:
        raise ValueError(f"The following cost elements are not valid column names: {invalid_cost}")


def bqtmizer(dataset_path: str, effectiveness_list: list, cost_list: list, token: str, weights: dict = None, N:int = 30, beta:float = 0.9):
    """
    Harnesses the power of quantum annealing (QA) to address large-scale real-world test case minimization (TCM) challenges.
    BQTmizer is designed to select the smallest possible test suite while ensuring that all testing objectives are met.
    As a hybrid solution, it integrates bootstrap sampling techniques to optimize qubit usage in QA hardware.

    Args:
        dataset_path (str): Path to the dataset, which should be a csv file. Each column represents a property, and each row represents a test case.

        effectiveness_list (list): List of effectiveness properties, should be valid column names in the dataset.

        cost_list (list): List of cost properties, should be valid column names in the dataset.

        token (str): Authentication token from the D-Wave Leap Platform.

        weights (dict, optional): Weights for the properties. None means equal weights. If not None, all properties
        inside effectiveness and cost list should be included. There should also be one extra key, "num", which represents
        the objective of minimizing the numer of test cases selected. All values should sum up to 1.

        N (int, optional): Size of sub-problems. Defaults to 30.

        beta (float, optional): Coverage percentage parameter in Bootstrap Sampling. Defaults to 0.9.

    Returns:
        result_df (dataframe): The solution of the test case minimization problem, containing all test cases selected.
    """


    warnings.filterwarnings('ignore')

    try:
        validated_path = validate_and_resolve_path(dataset_path)
        print(validated_path)
        df = pd.read_csv(validated_path)

        if weights is None:
            total_elements = len(effectiveness_list) + len(cost_list)
            default_weight = 1 / (total_elements+1)

            # Build the weights dictionary with equal values
            weights = {item: default_weight for item in (effectiveness_list + cost_list)}
            weights["num"] = default_weight

        validate_weights(weights, effectiveness_list, cost_list)

        column_names = set(df.columns)
        # Validate the effectiveness_list and cost_list
        validate_lists(effectiveness_list, cost_list, column_names)

        M = random_sample_unique(len(df), beta, N)
        print(str(M)+" sub-problems are generated.")
        sample_total_list = bootstrap_sampling(len(df), M, N)
        file_name = os.path.splitext(os.path.basename(validated_path))[0]
        # Quantum annealing results
        sample_first_list, qpu_access_list, qubit_avg, log_df, sampleset_list = run_qpu(
            sample_total_list, df, N, file_name, token, effectiveness_list, cost_list,
            weights)

        merge_sample = merge(sample_first_list)
        # sample_first_list, qpu_access_list, sum(qubit_num_list) / len(qubit_num_list)
        # sample, data, qpu_access_list, qubit_avg, cost_features, eff_features, weights
        result_df, sum_df = print_results(merge_sample, df, qpu_access_list, qubit_avg, cost_list, effectiveness_list,
                                          weights)

        # Get the directory of the calling file
        calling_file_dir = os.path.dirname(validated_path)
        if not os.path.exists(os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta))):
            os.makedirs(os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta)))
        if not os.path.exists(os.path.join(calling_file_dir, file_name+"_BQTmizer",str(N)+"_"+str(beta), "subproblem_execution_info")):
            os.makedirs(os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta), "subproblem_execution_info"))
        result_path = os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta), "result.csv")
        sum_path = os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta), "summmary.csv")
        log_path = os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta), "log.csv")

        result_df.to_csv(result_path)
        sum_df.to_csv(sum_path)
        log_df.to_csv(log_path)

        for sample_index in range(len(sampleset_list)):
            sampleset_list[sample_index].to_csv(os.path.join(calling_file_dir, file_name+"_BQTmizer", str(N)+"_"+str(beta), "subproblem_execution_info", "sample_"+str(sample_index)+".csv"))
        print("All files are generated under the path: "+os.path.join(calling_file_dir, file_name+"_BQTmizer",str(N)+"_"+str(beta)))
    except ValueError as e:
        print(f"Error: {e}")
        raise  # Propagate the error to the caller

    return result_df