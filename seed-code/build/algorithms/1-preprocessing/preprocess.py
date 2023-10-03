"""Feature engineers the customer churn dataset."""
import logging
import numpy as np
import pandas as pd
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    dir = "/opt/ml/processing/input"

    input_data_path = ""

    # We don't know the ordering of the folders/files in the input, but we know there is only one file.
    for file in os.listdir(dir):
        input_data_path = os.path.join("/opt/ml/processing/input", file)
        if os.path.isfile(input_data_path):
            input_data_path = os.path.join("/opt/ml/processing/input", file)
            break

    assert os.path.isfile(input_data_path)
    print(input_data_path)

    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")
        os.makedirs("/opt/ml/processing/train_data_with_headers")
    except:
        pass

    logger.info("Reading input dat2")

    # read csv
    df = pd.read_csv(input_data_path)

    # drop the "Phone" feature column
    df = df.drop(["Phone"], axis=1)

    # Change the data type of "Area Code"
    df["Area Code"] = df["Area Code"].astype(object)

    # Drop several other columns
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # Convert categorical variables into dummy/indicator variables.
    model_data = pd.get_dummies(df)

    # Create one binary classification target column
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )

    model_data = model_data.rename(columns={"Churn?_True.": "Churn_True"})

    # Split the data
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    train_data.to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
    train_data.to_csv(
        "/opt/ml/processing/train_data_with_headers/train.csv", header=True, index=False
    )
    validation_data.to_csv(
        "/opt/ml/processing/validation/validation.csv", header=False, index=False
    )
    test_data.to_csv("/opt/ml/processing/test/test.csv", header=False, index=False)

    train_data = train_data.drop(["Churn_True"], axis=1)
    train_data.to_csv(
        "/opt/ml/processing/data_baseline_with_headers/baseline.csv",
        header=True,
        index=False,
    )
