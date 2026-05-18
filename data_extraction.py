import os
import json
import pandas as pd

data = []

path = r"C:\Users\mansi\Desktop\ML\pulse-master\pulse-master\data\aggregated\transaction\country\india\state"

for state in os.listdir(path):
    state_path = os.path.join(path, state)

    for year in os.listdir(state_path):
        year_path = os.path.join(state_path, year)

        for file in os.listdir(year_path):

            if file.endswith(".json"):

                file_path = os.path.join(year_path, file)

                with open(file_path, "r") as f:

                    d = json.load(f)

                    if d["data"]["transactionData"] is not None:

                        for item in d["data"]["transactionData"]:

                            data.append({
                                "state": state,
                                "year": int(year),
                                "quarter": int(file.replace(".json", "")),
                                "type": item["name"],
                                "count": item["paymentInstruments"][0]["count"],
                                "amount": item["paymentInstruments"][0]["amount"]
                            })

# CREATE DATAFRAME
df = pd.DataFrame(data)

# PRINT CHECK
print(df.head())
print(df.shape)

# SAVE CSV
df.to_csv(r"C:\Users\mansi\Desktop\ML\phonepe_project\transactions.csv", index=False)

print("transactions.csv created successfully!")