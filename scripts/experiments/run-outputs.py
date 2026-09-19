import pandas as pd

cv = pd.read_csv('assets/scores_uid.csv')

cv.groupby('stationarity').mean(numeric_only=True)