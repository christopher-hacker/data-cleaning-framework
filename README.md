# data-cleaning-framework

This is a python-based tool to help quickly standardize messy data from multiple sources. It uses a configuration file in yaml format to define the input data, transformations, and output specifications. The framework supports various data cleaning operations with minimal code, including dropping rows, renaming columns, applying queries, and replacing values. Additionally, it allows for the use of custom preprocessors and cleaners.

## Features

- Transform input data from multiple sources with minimal code
- Automated logging of data cleaning operations for reproducibility
- Custom preprocessors and cleaners for specific data types
- Support for various data cleaning operations like dropping rows, renaming columns, applying queries, and replacing values
- Easy-to-use configuration file in yaml format

## Installation

This is currently in development and hasn't been published to PyPI yet. You can install it from the source code by using this repo in your poetry configuration.

```bash
poetry add git+ssh://git@github.com:christopher-hacker/data-cleaning-framework.git
```

## Usage

To use the data cleaning framework, you need to create a configuration file in yaml format. Here is an example configuration file:

```yaml
output_file: "cleaned_data.csv"
schema_file: "schemas/schema.py"
cleaners_file: "cleaners/cleaners.py"
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

You can then run the data cleaning framework using the following command:

```bash
dcf clean-data config.yaml
```

See [docs/config-options](docs/config-options.md) for a full list of configuration options.