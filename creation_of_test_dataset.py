import numpy
import pandas
import sklearn  
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


raw_df = pandas.read_csv(r"C:\Users\ASUS\Desktop\e\sklearn\logistic regression\weatherAUS.csv")

raw_df = raw_df.dropna(subset=["RainToday" , "RainTomorrow"] , inplace= True)

years = pandas.to_datetime(raw_df.Date).dt.year

train_df = raw_df[years < 2015]
val_df = raw_df[ years == 2015]
test_df = raw_df[years > 2015]

input_col =  list(train_df.columns)[1:-1]
target_col = "RainTomorrow"

train_input = train_df[input_col].copy()
train_target = train_df[target_col].copy()

val_input = val_df[input_col].copy()
val_target = val_df[target_col].copy()

test_input = test_df[input_col].copy()
test_target = test_df[target_col].copy()

numeric_col = train_input.select_dtypes(include= numpy.number).columns.tolist()
catagorical_col = train_input.select_dtypes("object").columns.tolist()

imputer = SimpleImputer(strategy="mean")

train_input[numeric_col] = imputer.transform(train_input[numeric_col])









