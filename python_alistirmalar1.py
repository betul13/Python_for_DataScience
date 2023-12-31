###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
###############################################

x = 8 #int


y = 3.2 #float


z = 8j + 18 #complex


a = "Hello World" #string


b = True #bool


c = 23 < 22 #bool



l = [1, 2, 3, 4,"String",3.2, False] #list



d = {"Name": "Jake",
     "Age": [27,56],
     "Adress": "Downtown"} #dict


t = ("Machine Learning", "Data Science") #tuple



s = {"Python", "Machine Learning", "Data Science","Python"} #set




###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

text = "The goal is to turn data into information, and information into insight."
text = text.upper().replace(",", " ").replace(".", " ").split()
print(text)


###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]

# Adım 1: Verilen listenin eleman sayısına bakın.

print(len(lst))

# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.

print((lst[0],lst[10]))

# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.

lst1 = lst[0:4]

# Adım 4: Sekizinci index'teki elemanı silin.

lst.pop(8)


# Adım 5: Yeni bir eleman ekleyin.

lst.insert(8,"o")

# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.

lst[8] = "N"

###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}


# Adım 1: Key değerlerine erişiniz.

print(dict.keys())

# Adım 2: Value'lara erişiniz.

print(dict.values())

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.

dict["Daisy"][1]=13

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.

dict["Ahmet"] = ["Turkey",24]

# Adım 5: Antonio'yu dictionary'den siliniz.

del dict["Antonio"]

###############################################
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
###############################################

l = [2,13,18,93,22]

def func(l):
    even_list = []
    odd_list = []
    for i in l:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

even_list, odd_list = func(l)
print(even_list, odd_list)

###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]
muhendısler = ogrenciler[0:3]
tıpcılar = ogrenciler[3:]

for index,value in enumerate(muhendısler):
      print(f"Mühendislik Fakültesi {index + 1}. öğrenci : {value}")
for index,value in enumerate(tıpcılar):
        print(f"Tıp Fakültesi {index + 1}. öğrenci : {value}")

###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

bilgi = list(zip(kredi,ders_kodu,kontenjan))
for a,b,c in bilgi:
        print(f"Kredisi {a} olan {b} kodlu dersin kontenjanı {c}")
#k= [f"Kredisi {a} olan {b} kodlu dersin kontenjanı {c}" for a,b,c in list(zip(kredi,ders_kodu,kontenjan))]
#print(k)
###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

if kume1.issuperset(kume2):
        print(kume1.intersection(kume2))
else :
        print(kume2.difference(kume1))
