import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import shuffle

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()

test.head()

train.isnull().sum()

test.isnull().sum()

#ALTERANDO O TIPO DAS COLUNAS REGION_CODE E POLICY_SALES_CHANNEL DE FLOAT PARA INT
vf_train = ['Region_Code', 'Policy_Sales_Channel']
vf_test = ['Region_Code', 'Policy_Sales_Channel']

def to_type(DataFrame, column, type):
    for col in column:
        DataFrame[col] = DataFrame[col].astype(type)

to_type(train, vf_train, 'int')
to_type(test, vf_test, 'int')

train.info()

test.info()

#PLOTANDO UM GRAFICO PARA VER A COLUNAS COM VALORES UNICOS(EX 1,2)
#COMEÇANDO PELAS COLUNAS DOS DADOS DE TRAIN
valores_uncios_train = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Response']

for f in valores_uncios_train:
    train[f].value_counts().plot(kind='bar')
    plt.title(f)
    plt.grid()
    plt.show()

#AGORA OS DADOS DE TEST
valores_uncios_test = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']

for e in valores_uncios_test:
    test[e].value_counts().plot(kind='bar')
    plt.title(e)
    plt.grid()
    plt.show()

#ALTERANDO OS VALOR YES OU NO PARA 1 OU 0 E CRIANDO UMA NOVA COLUNA PARA ESSES VALORES

def dano_veiculo(valor):
    if valor == 'Yes':
        return 1
    else:
        return 0

train['Vehicle_Dan'] = train['Vehicle_Damage'].map(dano_veiculo)
test['Vehicle_Dan'] = test['Vehicle_Damage'].map(dano_veiculo)


#ALTERANDO OS VALORES FEMALE E MALE PARA O E 1 NA COLUNA GENDER
train['Gender'] = train['Gender'].map( {'Female': 0, 'Male':1} ).astype(int)

train = pd.get_dummies(train, drop_first=True) 

#CRIANDO UMA NOVA COLUNA PARA OS CARROS COM 1 NO E OUTRA COLUNA PARA OS CARROS COM 2 ANOS NO DATASET TRAIN
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_1_Year','Vehicle_Age_2_Years','Region_Code','Policy_Sales_Channel']


train = train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_1_Year", "Vehicle_Age_< 2 Year": "Vehicle_Age_2_Years"})

train = train.drop("Vehicle_Damage_Yes", axis=1)

train.head()


#CRIANDO UMA NOVA COLUNA PARA OS CARROS COM 1 NO E OUTRA COLUNA PARA OS CARROS COM 2 ANOS NO DATASET TEST
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male':1} ).astype(int)

test = pd.get_dummies(test, drop_first=True) 

#CRIANDO UMA NOVA COLUNA PARA OS CARROS COM 1 NO E OUTRA COLUNA PARA OS CARROS COM 2 ANOS NO DATASET TRAIN
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_1_Year','Vehicle_Age_2_Years','Region_Code','Policy_Sales_Channel']

test = test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_1_Year", "Vehicle_Age_< 2 Year": "Vehicle_Age_2_Years"})

test.head()

variaveis = ['Gender', 'Driving_License', 'Previously_Insured', 'Vintage' ,'Vehicle_Dan', 'Vehicle_Age_1_Year', 'Vehicle_Age_> 2 Years']

#SEPARANDO OS DADOS EM TREINO E TESTE
X = train[variaveis]
y = train['Response']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.5)

modelo = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0) #CRIANDO O MODELO

modelo.fit(X_treino, y_treino) #TREINANDO O MODELO

m = modelo.predict(X_teste) #FAZENDO AS PREVISÕES DO MODELO

m

np.mean(y_teste == m)

#VALIDAÇÃO CRUZADA   

#modelo_antigo = 0.8773897299992128

resultado = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for linha_treino, linha_valid in kf.split(X):
    print("Treino: ", linha_treino.shape[0])
    print("Valid: ", linha_valid.shape[0])

    X_treino, X_teste = X.iloc[linha_treino], X.iloc[linha_valid]
    y_treino, y_teste = y.iloc[linha_treino], y.iloc[linha_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
    modelo.fit(X_treino, y_treino)

    m = modelo.predict(X_teste)

    acc = np.mean(y_teste == m)
    resultado.append(acc)
    print("ACC", acc)
    print()

np.mean(resultado)

#modelo_novo = 0.8774366387783845

#RETREINAR O MODELO

modelo = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
modelo.fit(X, y)

m = modelo.predict(test[variaveis])

sub = pd.Series(m, index=test['id'], name='Response')

sub