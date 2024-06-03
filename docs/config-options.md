# Configuration File Documentation

This document lists all the possible options that the `config.yaml` file can accept for the data cleaning framework.

## Root Options

- **output_file**: 
  - **Type**: string
  - **Required**: Yes
  - **Description**: Name of the output file where the cleaned data will be saved.
  - **Example**: `"cleaned_data.csv"`

- **input_files**:
  - **Type**: list of input file configurations
  - **Required**: Yes
  - **Description**: A list of input file configurations. Each input file can either be a direct file or use a preprocessor function.
  - **Example**:
    ```yaml
    input_files:
      - input_file: "data1.csv"
        sheet_name: "Sheet1"
        skip_rows: 1
        drop_rows: [0, 2, 4]
        assign_constant_columns:
          country: "USA"
        query: "column1 > 100"
        replace_values:
          column2:
            old_value: "new_value"
        rename_columns:
          column1: "new_column1"
          column2: "new_column2"
      - preprocessor:
          path: "scripts/preprocess.py"
          kwargs:
            param1: "value1"
    ```

- **schema_file**:
  - **Type**: string
  - **Required**: Yes
  - **Description**: The path to the schema file that will be used for validation.
  - **Example**: `"schemas/schema.py"`

- **assign_constant_columns**:
  - **Type**: dictionary of strings to any values
  - **Required**: No
  - **Description**: A dictionary of column names to values that will be assigned to all rows.
  - **Example**:
    ```yaml
    assign_constant_columns:
      source: "example_source"
    ```

- **cleaners_file**:
  - **Type**: string
  - **Required**: No
  - **Description**: The path to a cleaners file to use. Defaults to None. If not provided, will try to import the name 'cleaners' from the current namespace.
  - **Example**: `"cleaners/cleaners.py"`

## Input File Configuration Options

- **input_file**:
  - **Type**: string
  - **Required**: No
  - **Description**: Name of the input file to read from the input folder. Alternatively, you can provide a preprocessor function to read the file.
  - **Example**: `"data1.csv"`

- **preprocessor**:
  - **Type**: dictionary with path and kwargs
  - **Required**: No
  - **Description**: Details of the preprocessor function.
  - **Example**:
    ```yaml
    preprocessor:
      path: "scripts/preprocess.py"
      kwargs:
        param1: "value1"
    ```

- **sheet_name**:
  - **Type**: string
  - **Required**: No
  - **Description**: Name of the sheet to read from the Excel file. Defaults to the first sheet.
  - **Example**: `"Sheet1"`

- **skip_rows**:
  - **Type**: integer
  - **Required**: No
  - **Description**: Number of rows to skip when reading the Excel file.
  - **Example**: `1`

- **drop_rows**:
  - **Type**: list of integers
  - **Required**: No
  - **Description**: List of row indices to drop from the DataFrame after reading the file.
  - **Example**: `[0, 2, 4]`

- **assign_constant_columns**:
  - **Type**: dictionary of strings to any values
  - **Required**: No
  - **Description**: Dictionary of column names to values to assign to all rows in this specific file.
  - **Example**:
    ```yaml
    assign_constant_columns:
      country: "USA"
    ```

- **query**:
  - **Type**: string
  - **Required**: No
  - **Description**: Query to apply to the DataFrame after reading the file.
  - **Example**: `"column1 > 100"`

- **replace_values**:
  - **Type**: dictionary of strings to dictionaries
  - **Required**: No
  - **Description**: Dictionary of column names to dictionaries of values to replace.
  - **Example**:
    ```yaml
    replace_values:
      column2:
        old_value: "new_value"
    ```

- **rename_columns**:
  - **Type**: dictionary of integers or strings to strings
  - **Required**: No
  - **Description**: Dictionary of column names or indices to rename.
  - **Example**:
    ```yaml
    rename_columns:
      column1: "new_column1"
      column2: "new_column2"
    ```

## Preprocessor Configuration Options

- **path**:
  - **Type**: string
  - **Required**: Yes
  - **Description**: The path to a Python file containing a function called `preprocess` that returns a DataFrame.
  - **Example**: `"scripts/preprocess.py"`

- **kwargs**:
  - **Type**: dictionary of strings to any values
  - **Required**: No
  - **Description**: A dictionary of keyword arguments to pass to the `preprocess` function.
  - **Example**:
    ```yaml
    kwargs:
      param1: "value1"
    ```