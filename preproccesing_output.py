import numpy
import pandas
import sklearn  
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


raw_df = pandas.read_csv(r"C:\Users\ASUS\Desktop\e\sklearn\logistic regression\weatherAUS.csv")

raw_df.dropna(subset=["RainToday" , "RainTomorrow"] , inplace= True)

years = pandas.to_datetime(raw_df.Date).dt.year

# creates dataframes
train_df = raw_df[years < 2015]
val_df = raw_df[ years == 2015]
test_df = raw_df[years > 2015]

# define input columns
input_col =  list(train_df.columns)[1:-1]
target_col = "RainTomorrow"

# creates input dataframe
train_input = train_df[input_col].copy()
train_target = train_df[target_col].copy()

val_input = val_df[input_col].copy()
val_target = val_df[target_col].copy()

test_input = test_df[input_col].copy()
test_target = test_df[target_col].copy()

# seprates the number and object col in the dataframe
numeric_col = train_input.select_dtypes(include= numpy.number).columns.tolist()
catagorical_col = train_input.select_dtypes("object").columns.tolist()

# Logistic reggression model fails if we have NaN, se we need to fill that up
imputer = SimpleImputer(strategy="mean")

# fill the imputer data in the actual dataset (here we create what to fit)
imputer.fit(train_input[numeric_col])

# now its gonna fill all the NaN with the mean
train_input[numeric_col] = imputer.transform(train_input[numeric_col])
val_input[numeric_col] = imputer.transform(val_input[numeric_col])
test_input[numeric_col] = imputer.transform(test_input[numeric_col])

from sklearn.preprocessing import MinMaxScaler
# since the data ranges from very low very high in some cases, and that might create an issue
# so we scale all of them between (0,1) or sometimes between (-1,1)

scaler = MinMaxScaler()

# here it find min and max of each col
scaler.fit(raw_df[numeric_col])

# here we scales it
train_input[numeric_col] = scaler.transform(train_input[numeric_col])
val_input[numeric_col] = scaler.transform(val_input[numeric_col])
test_input[numeric_col] = scaler.transform(test_input[numeric_col])

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output= False , handle_unknown="ignore")

raw_df2 = raw_df[catagorical_col].fillna("unknown")
encoder.fit(raw_df2)

encoded_col = list(encoder.get_feature_names_out(catagorical_col))
# ====== ONE HOT ENCODING PROPER WAY (NO FRAGMENTATION) ======

# Transform
train_encoded = encoder.transform(train_input[catagorical_col].fillna("unknown"))
val_encoded = encoder.transform(val_input[catagorical_col].fillna("unknown"))
test_encoded = encoder.transform(test_input[catagorical_col].fillna("unknown"))

print("train input: ", train_encoded.shape)
print("train target: ", train_target.shape)
print("val input: ", val_encoded.shape)
print("val target: ", val_target.shape)
print("test input: ", test_encoded.shape)
print("testtarget: ", test_target.shape)

pandas.DataFrame(train_encoded).to_parquet("train_encoded.parquet")
pandas.DataFrame(val_encoded).to_parquet("val_encoded.parquet")
pandas.DataFrame(test_encoded).to_parquet("test_encoded.parquet")

pandas.DataFrame(train_target).to_parquet("train_target.parquet")
pandas.DataFrame(val_target).to_parquet("val_target.parquet")
pandas.DataFrame(test_target).to_parquet("test_target.parquet")

