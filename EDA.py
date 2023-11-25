import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",1000)

df = sns.load_dataset("titanic")
cat_cols = [col for col in df.columns if (str(df[col].dtypes)) in ["category","bool","object"]]
num_but_cat = [col for col in df.columns if (df[col].dtypes in ["int64","float64"]) and (df[col].nunique() < 10)]
print(num_but_cat)
cat_but_car = [col for col in df.columns if (str(df[col].dtypes) in ["category","object"]) and (df[col].nunique() > 20)]
cat_cols += num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car ]
num = [col for col in df.columns if col not in cat_cols and df[col].dtypes in ["int64","float64" ] ]
print(num)
#print(df[cat_cols])


def cat_summary (dataframe,col, plot = False):
   print(pd.DataFrame({col:dataframe[col].value_counts(),
                       "Ratio": 100 * dataframe[col].value_counts()/len(dataframe)}))
   print("#########################################################################")

   if plot:
       sns.countplot(x = dataframe[col],data=dataframe)
       plt.show(block = True) #birden fazla döngüye gireceğiz görseller için block = True karışıklığı önler.
cat_summary(df,"survived")

for col in cat_cols :
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")
        cat_summary(df,col,plot = True)

    else :
        cat_summary(df,col,plot = True)

def num_summary(dataframe,numerical,plot = False):
    quantiles = [0.25, 0.50, 0.75, 0.99]
    print(dataframe[numerical].describe(percentiles = quantiles).T)
    if plot :
        dataframe[numerical].hist()
        plt.xlabel(numerical)
        plt.title(numerical)
        plt.show(block = True)

for col in num :
    num_summary(df,col,plot= True)

