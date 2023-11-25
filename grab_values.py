import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",1000)

df = sns.load_dataset("titanic")

def grab_col_names(dataframe,cat_th = 10,car_th = 20) :
    cat_cols = [col for col in dataframe.columns if (str(dataframe[col].dtypes)) in ["category", "bool", "object"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int64", "float64"]) and (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object"]) and (dataframe[col].nunique() > car_th)]
    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if col not in cat_cols and dataframe[col].dtypes in ["int64","float64"]]

    print(f"Observation : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car = grab_col_names(df)

def cat_summary (dataframe,col, plot = False):
   print(pd.DataFrame({col:dataframe[col].value_counts(),
                       "Ratio": 100 * dataframe[col].value_counts()/len(dataframe)}))
   print("#########################################################################")

   if plot:
       sns.countplot(x = dataframe[col],data=dataframe)
       plt.show(block = True) #birden fazla döngüye gireceğiz görseller için block = True karışıklığı önler.

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

for col in num_cols:
    num_summary(df,col,plot=True)

#hedef değişken analizi

df.groupby("sex")["survived"].mean()
def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)

def target_summary_with_num(dataframe,target,num_cols):
    print(dataframe.groupby(target).agg({num_cols : "mean"}))

for col in num_cols:
    target_summary_with_num(df,"survived",col)