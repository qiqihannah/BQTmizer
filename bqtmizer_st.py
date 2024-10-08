import streamlit as st
import pandas as pd
from fractions import Fraction
import os
import zipfile
import io
from bootqa import random_sample_unique, bootstrap_sampling, run_qpu, merge, print_results


# A function to zip a list of DataFrames into a single ZIP file in memory
def create_zip(dataframes):
    zip_buffer = io.BytesIO()  # In-memory ZIP file buffer
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, (df, filename) in enumerate(dataframes):
            # Convert each DataFrame to CSV and write to ZIP
            csv_buffer = io.StringIO()  # Use StringIO for CSV in memory
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr(f"{filename}.csv", csv_buffer.getvalue())  # Naming each file in the ZIP
    zip_buffer.seek(0)  # Rewind the buffer to the beginning for reading
    return zip_buffer


# Function to convert DataFrame to CSV and return as bytes
def convert_df_to_csv(df):
    csv_buffer = io.StringIO()  # Use StringIO for CSV in memory
    df.to_csv(csv_buffer, index=False)  # Write DataFrame to CSV buffer
    return csv_buffer.getvalue().encode('utf-8')  # Return bytes-like object


# Title and description of the web app with enhanced formatting
st.markdown("""
# Welcome to **BQTmizer**! 
Our innovative tool harnesses the power of quantum annealing (QA) to address large-scale real-world test case minimization (TCM) challenges. **BQTmizer** is designed to select the smallest possible test suite while ensuring that all testing objectives are met. 

As a hybrid solution, it integrates bootstrap sampling techniques to optimize qubit usage in QA hardware. **BQTmizer** enhances your testing process, making it faster, smarter, and more efficient.
""")

# Step 1: Input for D-Wave API token (visible from the start)
st.sidebar.header("Configuration")
api_token = st.sidebar.text_input("Paste your D-Wave API token", type="password")

# Step 2: Input for subproblem size (integer)
subproblem_size = st.sidebar.number_input("Enter the subproblem size", min_value=1, step=1)

# Step 3: Input for coverage rate (float)
coverage_rate = st.sidebar.number_input("Enter the desired coverage rate (as a float)", min_value=0.0, max_value=1.0,
                                        step=0.01)

# Step 4: File uploader for the CSV (visible from the start)
uploaded_file = st.sidebar.file_uploader("Upload your test cases as a CSV file", type=["csv"])

# Ensure all inputs are available from the start
if uploaded_file is not None and api_token and subproblem_size and coverage_rate:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.markdown("### Uploaded CSV File:")
    st.dataframe(df, height=200)

    # Extract columns from the DataFrame
    columns = df.columns.tolist()

    # Step 5: Ask the user to categorize the columns
    st.markdown("### Please categorize the properties:")

    col1, col2 = st.columns(2)

    with col1:
        cost_features = st.multiselect(
            "Select the **cost properties** (e.g., time, resources used):", columns)
    with col2:
        effective_features = st.multiselect(
            "Select the **effective properties** (e.g., failure rate, code coverage):", columns)

    if cost_features and effective_features:
        # Join the list items into a single string for display
        formatted_cost_features = ", ".join(cost_features)
        formatted_effective_features = ", ".join(effective_features)

        st.markdown(f"**Cost Properties**: {formatted_cost_features}")
        st.markdown(f"**Effective Properties**: {formatted_effective_features}")

        # Step 6: Assign weights to each feature + test case number
        total_features = len(cost_features) + len(effective_features) + 1  # +1 for test case number
        default_weight = 1 / total_features

        st.markdown("### Specify the weights for each propertie:")
        st.markdown(
            "_Hint: You can input both decimals (e.g., `0.25`) or fractions (e.g., `1/4`). All weights must sum to 1._")

        # Create a dictionary to store the weights
        weights = {}


        # Function to convert input to float, allowing both decimal and fractions
        def convert_to_float(input_str):
            try:
                return float(Fraction(input_str))  # Converts fractions or decimals to float
            except ValueError:
                return None


        # Input for test case number weight
        test_case_weight_str = st.text_input(f"Weight for **test case number** (default {default_weight:.4f}):",
                                             value=str(default_weight))
        test_case_weight = convert_to_float(test_case_weight_str)
        weights['num'] = test_case_weight

        # Input for each cost feature
        for feature in cost_features:
            weight_str = st.text_input(f"Weight for cost propertie **{feature}** (default {default_weight:.4f}):",
                                       value=str(default_weight))
            weight = convert_to_float(weight_str)
            weights[feature] = weight

        # Input for each effective feature
        for feature in effective_features:
            weight_str = st.text_input(f"Weight for effective propertie **{feature}** (default {default_weight:.4f}):",
                                       value=str(default_weight))
            weight = convert_to_float(weight_str)
            weights[feature] = weight

        # Validate that all inputs are valid numbers
        if None in weights.values():
            st.error("Please enter valid numbers (decimal or fraction) for all weights.")
        else:
            # Ensure that the sum of the weights is exactly 1
            total_weight = sum(weights.values())
            st.markdown(f"**Total Weight**: {total_weight:.4f} _(must be 1.0)_")

            if abs(total_weight - 1.0) > 0.01:  # Allowing some floating point tolerance
                st.warning("The weights must add up to 1. Please adjust the weights accordingly.")
            else:
                # Step 7: Final execution with button click
                if st.button("Run Quantum Annealing"):
                    # Quantum annealing logic
                    subproblem_num = random_sample_unique(len(df), coverage_rate, subproblem_size)
                    sample_total_list = bootstrap_sampling(len(df), subproblem_num, subproblem_size)
                    file_name = os.path.splitext(uploaded_file.name)[0]
                    st.markdown(f"### {subproblem_num} sub-problems are generated.")

                    sample_first_list, qpu_access_list, qubit_avg, log_df, sampleset_list = run_qpu(
                        sample_total_list, df, subproblem_size, file_name, api_token, effective_features, cost_features,
                        weights)

                    merge_sample = merge(sample_first_list)
                    result_df, sum_df = print_results(merge_sample, df, qpu_access_list, qubit_avg, cost_features,
                                                      effective_features, weights)

                    # Save results in session state for persistence
                    st.session_state['log_df'] = log_df
                    st.session_state['result_df'] = result_df
                    st.session_state['sum_df'] = sum_df
                    st.session_state['sampleset_list'] = sampleset_list
                    st.session_state['subproblem_num'] = subproblem_num


# Show download buttons if quantum annealing results are present in session state
if 'log_df' in st.session_state:
    st.markdown("### Selected test cases in the final solution:")
    st.dataframe(st.session_state['result_df'])

    st.markdown(f"### {st.session_state['subproblem_num']} sub-problems are generated.")

    # Prepare files for download
    # dataframes_to_zip = st.session_state['sampleset_list']
    sampleset_with_name = []
    for sample_index in range(len(st.session_state['sampleset_list'])):
        sampleset_with_name.append((st.session_state['sampleset_list'][sample_index], "sample_"+str(sample_index)))
    # dataframes_to_zip = [
    #     (st.session_state['log_df'], "log"),
    #     (st.session_state['result_df'], "result"),
    #     (st.session_state['sum_df'], "summary")
    # ]

    zip_file_buffer = create_zip(sampleset_with_name)

    # Convert individual DataFrames to CSV for separate downloads
    csv_log = convert_df_to_csv(st.session_state['log_df'])
    csv_result = convert_df_to_csv(st.session_state['result_df'])
    csv_sum = convert_df_to_csv(st.session_state['sum_df'])

    # Download buttons
    st.download_button("Download ZIP of all CSVs", zip_file_buffer, file_name="subproblems_execution_info.zip",
                       mime="application/zip")
    st.download_button("Download log.csv", csv_log, file_name="log.csv", mime="text/csv")
    st.download_button("Download result.csv", csv_result, file_name="result.csv", mime="text/csv")
    st.download_button("Download summary.csv", csv_sum, file_name="summary.csv", mime="text/csv")
