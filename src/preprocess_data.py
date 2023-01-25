####
# DATA PREPROCESSING
# Create two DataFrames with these columns:
# (1) asin, num_reviews, review (one column per review)
# (2) asin, num_reviews, all_review (concatenated reviews in one column)
# Convert them to csv files for easy conversion back to DataFrame later.
####

import utils
import os
import pandas as pd

def process_data(dataframe):
    
    # DataFrame with columns 'asin', 'reviewText'
    # Header and First row:
    #   asin       reviewText          
    # 0 120401325X They look good and stick good!...
    
    df1 = dataframe[['asin','reviewText']]

    # List of unique asin
    # ['120401325X',...]
    
    product = df1['asin'].unique()

    # Convert to DataFrame
    # Header and First row:
    #   asin
    # 0 120401325X
    
    df_product = pd.DataFrame(product)
    df_product.columns = ['asin']
    
    review = []
    review_all = []

    # Add first asin to list
    #  ['120401325X']
    
    product = df1.loc[0]['asin']
    review.append(product)

    # Add all reviewText to each asin
    # [['120401325X', 'They look good and stick good!...','These stickers...',...],...]
    
    for i, row in df1.iterrows():
        if row['asin'] == product:
            review.append(row['reviewText'])
        else:
            review_all.append(review)
            review = []
            review.append(row['asin'])
            review.append(row['reviewText'])
            product = row['asin']
    review_all.append(review)    

    # Convert list to DataFrame
    # Header and First row:
    #   0          1                                 2
    # 0 120401325X They look good and stick good!... These stickers...
    
    df_review1 = pd.DataFrame(review_all)

    # Rename column header
    #   asin       1                                 2  
    # 0 120401325X They look good and stick good!... These stickers
    
    df_review1 = df_review1.rename(columns = {0:'asin'})
        
    # dataframe with columns 'asin', 'num_reviews'
    # Header and First row:
    #   asin        num_reviews 
    # 0 120401325X  5
    
    df2 = dataframe.asin.value_counts().reset_index()
    df2.columns = ['asin', 'num_reviews']

    # Add 'num_reviews' column to df_review1
    # Append a new empty column with header 'all_reviews'
    # Header and First row:
    #   asin       num_reviews 1                                 all_reviews
    # 0 120401325X 5           They look good and stick good!...
    
    df_review2 = pd.merge(df2, df_review1, on = 'asin')
    df_review2.loc[:,'all_reviews'] = ' '
    
    # Concatenate reviews into all_reviews
    # Create DataFrame with columns 'asin','num_reviews','all_reviews'
    #   asin       num_reviews  all_reviews
    # 0 120401325X 5            They look good and stick good!...
    
    for i, row in df_review2.iterrows():
        n = df_review2['num_reviews'][i]
        k = 1
        for k in range(1,n+1):
            df_review2['all_reviews'][i] = df_review2['all_reviews'][i] + ' ' + df_review2[k][i]
            k += 1
    df_review3 = df_review2[['asin','num_reviews','all_reviews']]
 
    return df_review2, df_review3

# RUN

DATA_DIR = "data/"
DATA_FILE = 'Cell_Phones_and_Accessories_5.json'

filepath = os.path.join(DATA_DIR, DATA_FILE)
df = utils.json_to_dataframe(filepath)

df1, df2 = process_data(df)

utils.dataframe_to_csv(df1,'data/asin_numreviews_review.csv')
utils.dataframe_to_csv(df2, 'data/asin_numreviews_allreview.csv')

# CHECK
file1 = os.path.join(DATA_DIR, 'asin_numreviews_review.csv')
file2 = os.path.join(DATA_DIR, 'asin_numreviews_allreview.csv')

df_review = utils.csv_to_dataframe(file1)
df_allreview = utils.csv_to_dataframe(file2)

print(len(df_review))
print(len(df_allreview))
print('')
print(df_review.loc[1556])
print('')
print(df_allreview.loc[1556])
print('')
print(df_allreview['all_reviews'][1556])