# Flight Delay Prediction and Disruption Risk Classification System

## Project overview
This academic project compares Logistic Regression and Random Forest for classifying whether a flight is delayed by more than 15 minutes. It includes data loading, basic feature preparation, Grid Search with cross-validation, evaluation metrics, feature importance where supported, and a Streamlit dashboard.

## Folder setup
Place your Kaggle CSV file inside the `data` folder. Keep only the CSV(s) you intend to use there. The scripts automatically select the largest CSV in that folder.

## Windows setup
1. Install Python 3.10 or later from the official Python website. During installation, tick **Add Python to PATH**.
2. Install Visual Studio Code and open this project folder.
3. Open **Terminal > New Terminal** in VS Code.
4. Run:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Check the dataset:
   ```bash
   python inspect_data.py
   ```
6. Train the models:
   ```bash
   python train.py
   ```
7. Start the dashboard:
   ```bash
   python -m streamlit run app.py
   ```

## Large dataset note
The training script reads up to 150,000 rows for the initial run so it is more likely to work on a student laptop with a large CSV. It does not read the whole 2 GB file into memory. The selected sample and metrics should be described honestly in the report.

## Dataset compatibility
The script looks for common departure-delay or arrival-delay target columns. If it cannot identify a target column, run `python inspect_data.py` and adapt the target column in `train.py` based on the actual dataset columns. Do not claim model results until training finishes successfully.

## Important academic note
The output is an academic prediction, not an operational airline system. A model trained on historical data can be biased or inaccurate, and columns recorded after a flight's departure/arrival should not be used as input features for a real pre-flight prediction. Review the feature columns and target definition with your instructor.
