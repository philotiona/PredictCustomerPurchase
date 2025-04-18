# Customer Purchase Prediction

This project aims to predict customer purchases based on historical data. By analyzing previous buying patterns, we can forecast future purchases, helping retailers optimize their inventory and better understand customer needs.

## Project Structure

- **data/**: Contains raw and processed data files.
  - **raw/**: Holds the original dataset.
    - `large_purchases.csv`: Raw data of customer purchases.
  - **processed/**: Intended for cleaned and transformed data files.

- **models/**: Contains model-related files.
  - `__init__.py`: Initializes the models package.

- **notebooks/**: Jupyter notebooks for data exploration and model development.
  - `01_data_exploration.ipynb`: For exploring the dataset and visualizing data distributions.
  - `02_model_development.ipynb`: For developing and training machine learning models.

- **src/**: Source code for data processing, modeling, and utilities.
  - `__init__.py`: Initializes the src package.
  - `data_preprocessing.py`: Functions for cleaning and transforming raw data.
  - `model.py`: Implementation of machine learning models.
  - `predict.py`: Functions for making predictions with trained models.
  - `utils.py`: Utility functions for various tasks.

- **tests/**: Contains unit tests for the project.
  - `__init__.py`: Initializes the tests package.
  - `test_utils.py`: Unit tests for utility functions.

- **config.json**: Configuration settings for the project, including file paths and model parameters.

- **requirements.txt**: Lists the required Python packages and their versions.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd customer-purchase-prediction
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Use the Jupyter notebooks in the `notebooks/` directory for data exploration and model development.
- Run the scripts in the `src/` directory for data processing and predictions.
- Modify the `config.json` file to adjust settings as needed.

## License

This project is licensed under the MIT License. See the LICENSE file for details.