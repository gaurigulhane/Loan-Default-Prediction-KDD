# Datasets

Download the following CSV files from their original sources and upload them to the Colab runtime (`/content`), or put them in Google Drive and set `MOUNT_DRIVE = True` and `DATA_DIR` in the notebook.

| Dataset | Expected CSV | Source | Target |
|---|---|---|---|
| Customer Behavior | `Training Data.csv` | https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior | `Risk_Flag` |
| Lending Club | `LC_loans_granting_model_dataset.csv` | https://zenodo.org/records/11295916 | `Default` |
| Credit Risk | `credit_risk_dataset.csv` | https://www.kaggle.com/datasets/laotse/credit-risk-dataset | `loan_status` |

The raw CSVs are not redistributed here because of file size and source-specific licensing/redistribution considerations. Check each source's terms before redistributing. The notebook will prompt for missing files in Colab.
