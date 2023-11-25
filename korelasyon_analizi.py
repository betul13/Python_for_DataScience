import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = pd.read_csv(r"C:\Users\bett0\Desktop\breast_cancer.csv")
df = df.iloc[:,1:-1] #istemediğimiz değişkenlerden kurtuluyoruz

num_cols = [col for col in df.columns if (df[col].dtypes in ["int64","float64"]) & (df[col].nunique() > 10) ]

corr = df[num_cols].corr()

sns.set(rc = {"figure.figsize":(12,12)})
sns.heatmap(corr,cmap = "RdBu") #ısı haritası çıkarılır
plt.show()

#yüksek korelasyonlu değişkenlerin silinmesi


cor_matrix = df.corr().abs()
upper_triangle_matrix =cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k = 1).astype(np.bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
df.drop(drop_list,axis = 1)
def high_correlated_cols(dataframe,plot=False,corr_th = 0.90):
    corr = dataframe.corr()
    cor_matrix =corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc = {"figure.figsize":(15,15)})
        sns.heatmap(corr,cmap="RdBu")
        plt.show()
    return drop_list
high_correlated_cols(df)
drop_list = high_correlated_cols(df,plot=True)
df.drop(drop_list,axis=1)
(high_correlated_cols(df.drop(drop_list,axis=1),plot=True))
