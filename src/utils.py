import json
import pandas as pd

def json_to_dataframe(filepath):
    all_reviews = []

    with open(filepath) as json_file:
        for j in json_file:
            x = json.loads(j)
            all_reviews.append(x)
        
    return pd.DataFrame(all_reviews)


def csv_to_dataframe(filepath):
    df = pd.read_csv(filepath)
    return df


def dataframe_to_csv(dataframe, filename):
    csvfile = dataframe.to_csv(filename, encoding='utf-8', index=False)