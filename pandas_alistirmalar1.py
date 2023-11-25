##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
print(df["sex"].value_counts())

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

print({col : df[col].nunique() for col in df.columns})

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

print(df["pclass"].unique())

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

print(df[["pclass","parch"]].nunique())

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

print(df["embarked"].dtypes)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtypes)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

print(df[df["embarked"]== "C"])

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

print(df[df["embarked"] != "S"])

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

print(df[(df["age"]< 30) & (df["sex"] == "male")])

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

print(df[(df["fare"] > 500) & (df["age"] > 70)])

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

print(df.isnull().sum())


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who",axis = 1 ,inplace=True)
print(df.columns)

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

mode_value = df['deck'].mode()[0] #ilk mode değeriyle dönüştürür
df['deck'].fillna(mode_value, inplace=True)
print(df["deck"])

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

median_age = df["age"].median()
df["age"].fillna(median_age,inplace=True)
print(df["age"])

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

print(df.groupby(["pclass","sex"])["survived"].agg(["sum","count","mean"]))

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

df['age_flag'] = df['age'].apply(lambda age: 1 if age < 30 else 0)
print(df)

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df1 = sns.load_dataset("tips")
print(df1.head())


#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

print(df1.groupby("time")["total_bill"].agg(["sum","min","max","mean"]))

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

print(df1.groupby(["time","day"])["total_bill"].agg(["sum","min","max","mean"]))

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

print(df1[(df1['time'] == 'Lunch') & (df1['sex'] == 'Female')].groupby("day")["tip"].agg(["sum","min","max","mean"]))

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

print(df1[(df1["size"] < 3) & (df1["total_bill"] > 10)].mean(numeric_only=True))

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df1["total_bill_tip_sum"] = df1["total_bill"]+ df1["tip"]
print(df1)


#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

top_30 = df1.sort_values(by = "total_bill_tip_sum",ascending= False).head(30).copy()
print(top_30)

