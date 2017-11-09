
# coding: utf-8

# In[1]:

#!/usr/bin/python

### External libraries
import sys
import pickle
import time
import copy

#sys.path.append("../tools/")
#sys.path.append("C:\\Users\\ubirajara.theodoro\\Dropbox\\jupyter_notebook\\ud120-projects-master\\tools\\")
#sys.path.append("C:\\Users\\ubirajara.schier\\Dropbox\\jupyter_notebook\\ud120-projects-master\\tools\\")
from feature_format import featureFormat, targetFeatureSplit

#sys.path.append("C:\\Users\\ubirajara.theodoro\\Dropbox\\jupyter_notebook\\ud120-projects-master\\final_project\\")
#sys.path.append("C:\\Users\\ubirajara.schier\\Dropbox\\jupyter_notebook\\ud120-projects-master\\final_project\\")
from tester import dump_classifier_and_data
from tester import test_classifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

# Import numpy, matplotlib and pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import scipy.stats as stats



# In[2]:

### > Internal Functions

# ----------------------------------------------
# Outlier removal using IQR rule
# ----------------------------------------------
def removefeatureIQRoutliers(dataset, item, percsup, percinf):

    qsup, qinf = np.percentile(dataset[item], [percsup , percinf])
    iqr = qsup - qinf
    new_min = qinf - (iqr*1.5)
    new_max = qsup + (iqr*1.5)
  
    dataset.loc[dataset[item]<new_min, ['isoutlier']] = True
    dataset.loc[dataset[item]>new_max, ['isoutlier']] = True
    
    outliers = dataset.loc[dataset['isoutlier'] == True, item]
    dataset = dataset.query('isoutlier==False')
    
    return dataset, outliers

# ----------------------------------------------
# Exibe correlações maiores que o valor desejado
# ----------------------------------------------
def printcorrelations(df, mincorr):
    print "> Correlações maiores que: ", mincorr
    dfcorr = df.corr()
    dfcorr = dfcorr[dfcorr >= mincorr].stack().reset_index()
    dfcorr = dfcorr.query('level_0 != level_1')
    dfcorr = dfcorr.sort_values(['level_0', 'level_1'], ascending=[True, True])
    return dfcorr

# ---------------------------------------------------------------------
# Plota gráfico de dispersão entre 2 atributos agrupados por POI/NO-POI
# ---------------------------------------------------------------------
def plotcorrelations(var_x, var_y, dataset):
    colors = {True:'red', False:'blue'}
    #dsname =[x for x in globals() if globals()[x] is dataset][0]
    #print "> Dataset:", dsname
    plt.figure()
    dataset.plot.scatter(x=var_x, y=var_y, c=dataset['poi'].apply(lambda x: colors[x]))
    plt.xlabel(var_x, fontsize=14)
    plt.ylabel(var_y, fontsize=14)
    plt.show()

# ---------------------------------------
# Plota histogramas das colunas numéricas
# ---------------------------------------
def plotviewfeatures(dataset, feature_list, withoutliers, percsup, percinf):

    outliers = pd.Series()
    dataset['isoutlier'] = False
    ds_tmp = dataset.copy()
    #ds_tmp = ds_tmp._get_numeric_data()

    for listitem in feature_list:
        if (listitem[1]==True and (listitem[3]=="n" or listitem[3]=="m" or listitem[3]=="p" or listitem[3]=="t")):
            print ""
            print "----------------------------------------------------------------------------------"
            print "> Atributo: ", listitem[0]
            print "----------------------------------------------------------------------------------"

            if (withoutliers==True and listitem[5]):
                ds_tmp, outliers = removefeatureIQRoutliers(ds_tmp, listitem[0], percsup, percinf)

            if (listitem[3]=="p"):
                xbins = np.linspace(0, 100, 20)
                ds_tmp.hist(column=listitem[0], by='poi', bins=xbins, sharex=True, sharey=True, align='mid')
                plt.xlim(0, 100)
            else:
                ds_tmp.hist(column=listitem[0], by='poi', bins=20, sharex=True, sharey=True, align='mid')
                plt.xlim(ds_tmp[listitem[0]].min(), ds_tmp[listitem[0]].max()*1.1)
                plt.ticklabel_format(style = 'plain')

            plt.xlabel(listitem[0], fontsize=14)
            plt.ylabel('Ocorrencias', fontsize=14)
           
            plt.show()
            plt.close()

            plt.figure()
            plotcorrelations("total_payments", listitem[0], ds_tmp)
            #sns.boxplot(x=ds_tmp[listitem[0]])
            plt.show()
            plt.close()

            if not outliers.empty:
                print "> Outliers removidos (IQR rule=", percsup, "% /", percinf, "%):"
                print outliers
                outliers = pd.Series()
            print ""
            print ">", "Observações: "
            if (listitem[4]!=""):
                print "-", listitem[4]
            print ds_tmp[listitem[0]].describe()
    return ds_tmp

# --------------------
# Atualiza POI Dataset
# --------------------
# seleciona os POI a partir de uma cópia do dataset original, mantendo apenas as colunas desejadas
# e excluindo os registros NaN
def f_list2ds(dataset, onlypoi, feature_list):
    df_new = dataset.copy()
    if (onlypoi == True):
        df_new = df_new.query('poi == True')
    for listitem in feature_list:
        if (listitem[1] == False):
            df_new = df_new.drop(listitem[0], axis=1)
    df_new = df_new.dropna(axis=0, how='any')
    df_new['isoutlier']=False
    return df_new

# ------------------------------------------------------------------
# atualiza a feature_list a partir das listas de trabalho utilizadas
# ------------------------------------------------------------------
def templst2featurelist(work_list, feature_list):
    for listitem in work_list:
        # Insere apenas os atributos selecionados e que existem no data_dict ("d"=default)
        if ((listitem[1] == True) and (listitem[2]=="d")):
            if not(any(listitem[0] in sublist for sublist in feature_list)):
                feature_list.append(listitem[0])
    return feature_list

# ---------------------------
# Insere uma feature na lista
# ---------------------------
def insertfeature(item, classe, tipo, feature_list, removeouliers):
    if not(any(item in sublist for sublist in feature_list)):
        feature_list.append([ item, True, classe, tipo, "", removeouliers])
    else:
        for listitem in feature_list:
            if (listitem[0] == item):
                listitem[1] = True
    return feature_list

# ------------------------------------
# Insere uma observação em uma feature
# ------------------------------------
def insertobsfeature(item, feature_list, obs):
    if (any(item in sublist for sublist in feature_list)):
        for listitem in feature_list:
            if (listitem[0] == item):
                listitem[4] = obs
    return feature_list

# ------------------------------------
# zera observações de todas as features
# ------------------------------------
def zeraobsfeatures(feature_list):
    for listitem in feature_list:
        listitem[4] = ""
    return feature_list

# ---------------------------
# Remove uma feature da lista
# ---------------------------
def removefeature(item, feature_list):
    if (any(item in sublist for sublist in feature_list)):
        for listitem in feature_list:
            if (listitem[0] == item):
                listitem[1] = False
    return feature_list

# ----------------------------------------
# Exibe lista com as features selecionadas
# ----------------------------------------
def printfeatures(feature_list):
    for listitem in feature_list:
        if (listitem[1] == True):
            print listitem[0]

# ---------------------------------------------------------
# Pré-seleciona as features com % de NaN < ou - ao desejado
# ---------------------------------------------------------
def selectPOIfeatures_percNaN(dataset, ispoi, feature_list, maxpercNaN):
    print '> Features selecionadas com % de NaN < ou = a', maxpercNaN, '%:'
    ds = dataset.copy()

    if (ispoi == True):
        ds = ds.query('poi == True')
    else:
        ds = ds.query('poi == False')
    totalreg = len(ds)

    for listitem in feature_list:
        #print listitem[0]
        perccount_nan = round((float(totalreg - ds[listitem[0]].count())/totalreg)*100,2)
        #print perccount_nan
        if (perccount_nan>maxpercNaN):
            listitem[1] = False
        else:
            listitem[1] = True

        if (listitem[1] == True):
            print '  - ', listitem[0], perccount_nan, '%', ' > feature selecionada'
        else:
            print '  - ', listitem[0], perccount_nan, '%', ' > feature descartada' 
            
    return feature_list




# In[3]:

def evaluate_model(clf, features, labels, cv):

    nested_score = cross_val_score(clf, X=features, y=labels, cv=cv, n_jobs=2)
    print "Nested f1 score: {}".format(nested_score.mean())

    clf.fit(features, labels)    
    print "Best parameters: {}".format(clf.best_params_)

    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    
    for train_index, test_index in cv.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf.best_estimator_.fit(X_train, y_train)
        pred = clf.best_estimator_.predict(X_test)

        cv_accuracy.append(accuracy_score(y_test, pred))
        cv_precision.append(precision_score(y_test, pred))
        cv_recall.append(recall_score(y_test, pred))
        cv_f1.append(f1_score(y_test, pred))

    print "Mean Accuracy: {}".format(np.mean(cv_accuracy))
    print "Mean Precision: {}".format(np.mean(cv_precision))
    print "Mean Recall: {}".format(np.mean(cv_recall))
    print "Mean f1: {}".format(np.mean(cv_f1))


# # Parte I - Data Wrangling
# # Passo 1: Carga Inicial dos Dados

# - O primeiro passo adotado foi importar o conjunto de dados final_project_dataset.pkl fornecido e convertê-lo para um dataframe pandas, a fim de facilitar a preparação dos dados e tambem a análise exploratória dos mesmos. 
# 
# 
# - Em uma primeira análise do dataset já convertido, será verificado o número total de pessoas, o número total de pessoas envolvidas na fraude, o número total de pessoas não envolvidas na fraude e, também, o número de atributos (features) disponíveis no dataset para as análises seguintes.

# In[4]:

### Carga do DataSet Inicial e das listas de atributos

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### > Converting to Pandas Dataframe
df = pd.DataFrame(data_dict.items(), columns=['name', 'features'])
df.set_index('name')

# expand df.listcol into its own dataframe
tags = df['features'].apply(pd.Series)

# rename each variable is listcol
tags = tags.rename(columns = lambda x : str(x))

# join the tags dataframe back to the original dataframe
df = pd.concat([df[:], tags[:]], axis=1)

# deleting old column with all features
del df['features']
    
### > Adjusting column types

# rótulo POI: [‘poi’] (atributo objetivo lógico (booleano), representado como um inteiro)

# atributos financeiros: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (todos em dólares americanos (USD))
df['salary'] = df['salary'].astype('float64', errors='ignore')
df['deferral_payments'] = df['deferral_payments'].astype('float64', errors='ignore')
df['total_payments'] = df['total_payments'].astype('float64', errors='ignore')
df['loan_advances'] = df['loan_advances'].astype('float64', errors='ignore')
df['bonus'] = df['bonus'].astype('float64', errors='ignore')
df['restricted_stock_deferred'] = df['restricted_stock_deferred'].astype('float64', errors='ignore')
df['deferred_income'] = df['deferred_income'].astype('float64', errors='ignore')
df['total_stock_value'] = df['total_stock_value'].astype('float64', errors='ignore')
df['expenses'] = df['expenses'].astype('float64', errors='ignore')
df['exercised_stock_options'] = df['exercised_stock_options'].astype('float64', errors='ignore')
df['other'] = df['other'].astype('float64', errors='ignore')
df['long_term_incentive'] = df['long_term_incentive'].astype('float64', errors='ignore')
df['restricted_stock'] = df['restricted_stock'].astype('float64', errors='ignore')
df['director_fees'] = df['director_fees'].astype('float64', errors='ignore')

# atributos de email: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (as unidades aqui são geralmente em número de emails; a exceção notável aqui é o atributo ‘email_address’, que é uma string)
df['to_messages'] = df['to_messages'].astype('float64', errors='ignore')
df['email_address'] = df['email_address'].astype('str', errors='ignore')
df['from_poi_to_this_person'] = df['from_poi_to_this_person'].astype('float64', errors='ignore')
df['from_messages'] = df['from_messages'].astype('float64', errors='ignore')
df['from_this_person_to_poi'] = df['from_this_person_to_poi'].astype('float64', errors='ignore')
df['shared_receipt_with_poi'] = df['shared_receipt_with_poi'].astype('float64', errors='ignore')
df['isoutlier'] = False
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# [0] : nome do atributo
# [1] : atributo selecionado (true/false)
# [2] : origem do atributo ('d'efault/'c'ustom)
# [3] : tipo do atributo ('n'umeric, 'b'oolean, 'm'oney, 'c'ategorical, 'd'iscrete, 'p'ercentual)
# [4] : observação para ser exibida abaixo de cada plotagens
# [5] : Remove outliers (Sim ou não)
features_list = ['poi']
poi_features  = [[ 'poi', True, 'd', 'b', "", False ],
                 [ 'name', True, 'c', 'c', "", False ],
                 [ 'salary', False, 'd', 'p', "", False ],
                 [ 'deferral_payments', False, 'd', 'p', "", False ],
                 [ 'total_payments', False, 'd', 't', "", True ],
                 [ 'loan_advances', False, 'd', 'p', "", False ],
                 [ 'bonus', False, 'd', 'p', "", False ],
                 [ 'restricted_stock_deferred', False, 'd', 'm', "", False ],
                 [ 'deferred_income', False, 'd', 'p', "", False ],
                 [ 'total_stock_value', False, 'd', 't', "", True ],
                 [ 'expenses', False, 'd', 'p', "", False ],
                 [ 'exercised_stock_options', False, 'd', 'm', "", False ],
                 [ 'other', False, 'd', 'p', "", False ],
                 [ 'long_term_incentive', False, 'd', 'p', "", False ],
                 [ 'restricted_stock', False, 'd', 'm', "", False ], 
                 [ 'director_fees', False, 'd', 'p', "", False ],
                 [ 'to_messages', False, 'd', 'n', "", False ],
                 [ 'email_address', True, 'c', 'c', "", False ],
                 [ 'from_poi_to_this_person', False, 'd', 'n', "", False ],
                 [ 'from_messages', False, 'd', 'n', "", False ],
                 [ 'from_this_person_to_poi', False, 'd', 'n', "", False ],
                 [ 'shared_receipt_with_poi', False, 'd', 'n', "", False ],
                 [ 'isoutlier', True, 'c', 'b', "", False ]]

#nopoi_features  = copy.deepcopy(poi_features)


print "> Observações:"
print "- Número de registros original do dataset:", len(df)
x = df.poi.value_counts()
print "- Número de pessoas envolvidas na fraude:", x[1]
print "- Número de pessoas não envolvidas na fraude:", x[0]
print "- Número de atributos do dataset:", len(df.columns)
print ""
print "> Amostragem do Dataset original (exibindo apenas os 5 primeiros registros):"
pd.set_option('display.max_columns', 30)
df.head(5)


# # Passo 2: Tratamento para Missing Values (NaN):
# 
# - Durante a fase exploratória dos dados, verificou-se que as colunas total_payments e total_stock_value são colunas que representam a soma dos demais atributos existentes. Entretanto, quando por exemplo uma pessoa não possui nenhuma despesa (expenses=0), o valor existente é NaN e não 0 (zero). O mesmo ocorre para todos os demais atributos financeiros. Conclui-se desta forma que se torna apropriado, para preservação dos dados, substituir estes atributos financeiros iguais à NaN por 0 (zero).
# 
# 
# - Já quanto aos atributos referente ao levantamento de envio e recebimento de e-mails, não se pode afirmar se quando é informado NaN é por que não existe a informação, ou se deveria ser 0 (zero). Partindo do princípio que a etapa de extração destes dados deve ter sido consistente com a regra identificada no item acima, acredita-se que atribuir o valor 0 (zero) à estes atributos seja mais apropriado do que atribuir qualquer outro valor, como por exemplo a média.

# In[5]:

df.fillna(0, inplace=True)
print "> Observações:"
print "- Alterado para 0, valores iguais à NaN"


# # Passo 3: Pré-seleção dos Atributos de Interesse

# - Nesta etapa será executado uma pré-seleção dos atributos disponíveis em função de um determinado percentual de missing values admissível.
# 
# 
# - Este procedimento é executado em duas etapas: a primeira etapa ocorre antes de se estabelecer os critérios para tratamento dos registros com valores inválidos, a fim de verificar quais são os atributos efetivamente disponíveis que atendam ao percentual desejado; Com base nestes resultados verifica-se a perda de informação e então inicia-se o processo para analisar regras que possam tratar os registros inválidos de forma a preservar os dados para as análises.
# 
# 
# - Abaixo os resultados encontrados para o dataset sem tratamento para os registros inválidos (ver passo 2). Como pode-se observar abaixo, a perda de registros e atributos com mais de 10% de registros inválidos é significativa, fazendo-se necessário tratar os dados de forma que o máximo de registros e atributos do dataset possam ser utilizados.
# 
# > Features selecionadas com % de NaN < ou = a 10 %:
#   -  poi 0.0 %  > feature selecionada
#   -  name 0.0 %  > feature selecionada
#   -  salary 5.56 %  > feature selecionada
#   -  deferral_payments 72.22 %  > feature descartada
#   -  total_payments 0.0 %  > feature selecionada
#   -  loan_advances 94.44 %  > feature descartada
#   -  bonus 11.11 %  > feature descartada
#   -  restricted_stock_deferred 100.0 %  > feature descartada
#   -  deferred_income 38.89 %  > feature descartada
#   -  total_stock_value 0.0 %  > feature selecionada
#   -  expenses 0.0 %  > feature selecionada
#   -  exercised_stock_options 33.33 %  > feature descartada
#   -  other 0.0 %  > feature selecionada
#   -  long_term_incentive 33.33 %  > feature descartada
#   -  restricted_stock 5.56 %  > feature selecionada
#   -  director_fees 100.0 %  > feature descartada
#   -  to_messages 22.22 %  > feature descartada
#   -  email_address 0.0 %  > feature selecionada
#   -  from_poi_to_this_person 22.22 %  > feature descartada
#   -  from_messages 22.22 %  > feature descartada
#   -  from_this_person_to_poi 22.22 %  > feature descartada
#   -  shared_receipt_with_poi 22.22 %  > feature descartada
#   -  isoutlier 0.0 %  > feature selecionada
# 
# > Observações:
# - Número de registros antes da seleção dos atributos: 146
# - Número de registros após a seleção dos atributos: 75
# 
# - Entretanto, como verificou-se que a maioria dos registros inválidos ocorrem em circustâncias que puderam ser tratadas (ver Passo 2), todos os atributos do dataset foram assim pré-selecionados. Eventualmente, quando não é possível tratar todas as circunstâncias em que ocorrem os missing values, adota-se como regra descartar os atributos cujo percentual de registros inválidos seja maior que o desejado.

# In[6]:

### > Pré-seleção das Features (cleaning)
df_poi = f_list2ds(df, False, poi_features)
numreg = len(df_poi)
# seleciona as features com % de NaN < ou = ao percentual desejado
print "> Dataset POI:"
print "- Seleciona os atributos a partir do dataset original em função do % de registros inválidos (NaN)"
print ""
poi_features = selectPOIfeatures_percNaN(df, True, poi_features, 10)
df_poi = f_list2ds(df, False, poi_features)
print ""
print "> Observações:"
print "- Número de registros antes da seleção dos atributos:", numreg
print "- Número de registros após a seleção dos atributos:", len(df_poi)


# # Passo 4. Limpeza de dados

# - Nesta etapa procuramos tratar os registros considerados inconsistentes, ou sejam, que apresentam valores mas os mesmos não estão corretos.
# 
# 
# - Neste processo, como foi identificado a regra em que os atributos total_payments e total_stock_value representam o somatório de outros atributos existentes no dataset, foi possível identificar os atributos que compunham o somatório se comportam de 3 maneiras:
# 
# > Atributos que compunham o somatório do total_payments:
# - Atributos que obrigatoriamente são caracterizados como deduções e, portanto, necessitam apresentar um valor menor que zero;
# - Atributos que obrigatoriamente não podem ser caracterizados como deduções e, portanto, necessitam apresentar um valor maior que zero;
# 
# > Atributos que compunham o somatório do total_stock_value:
# - Atributos voláteis, que ora se caracterizam como deduções e ora não, ou seja, como não se tem um conhecimento da regra de negócios, considera-se que podem assumir valores positivos ou negativos.
# 
# 

# In[7]:

print "> Identificando registros inválidos:"
print "- Nome=TOTAL"
df[df.name == 'TOTAL']


# In[8]:

print "> Identificando registros inválidos:"
print "- deferral_payments<0"
df[df.deferral_payments<0]


# In[9]:

print "> Identificando registros inválidos:"
print "- bonus<0"
df[df.bonus<0]


# In[10]:

print "> Identificando registros inválidos:"
print "- salary<0"
df[df.salary<0]


# In[11]:

print "> Identificando registros inválidos:"
print "- loan_advances<0"
df[df.loan_advances<0]


# In[12]:

print "> Identificando registros inválidos:"
print "- expenses<0"
df[df.expenses<0]


# In[13]:

print "> Identificando registros inválidos:"
print "- other<0"
df[df.other<0]


# In[14]:

print "> Identificando registros inválidos:"
print "- long_term_incentive<0"
df[df.long_term_incentive<0]


# In[15]:

print "> Identificando registros inválidos:"
print "- director_fees<0"
df[df.director_fees<0]


# In[16]:

print "> Identificando registros inválidos:"
print "- deferred_income>0"
df[df.deferred_income>0]


# In[17]:

df = df[df.name != 'TOTAL']
df = df[df.deferral_payments>=0]

print "> Observações:"
print "- Removido somente o registro referente pessoa com nome igual à TOTAL"
print "- Removido registros onde o campo deferral_payments é menor que zero"
print "- Número de registros após a limpeza:", len(df)


# # Passo 5: Alteração e/ou Criação de Atributos

# - Nesta etapa tratamos da criação de novos atributos e das alterações/transformações vistos como necessárias aos atributos existentes, conforme relacionado abaixo.
# 
# 
# - Entre as alterações efetuadas, destaca-se o recálculo do campo total_payments desconsiderando o atributo deferred_income. Esta fez-se necessária pois o atributo deferred_income é uma dedução e apresenta valores menores que zero. Ao removê-lo da soma junto com os demais atributos financeiros, foi possível transformar cada atributo financeiro de valor nominal para percentual em relação ao novo total_payments. Concluiu-se que esta alteração é essencial para análisar os dados, pois se torna muito mais importante analisar a parcitipação de cada atributo no total dos pagamentos (total_payments) do que analisar seu valor nominal em si. Por exemplo: O valor do bonus de 200.000 de uma determinada pessoa não nos diz tanto quanto saber que esse valor representou 80% do montante total pago à essa mesma pessoa quando a média deste percentual foi de, por exemplo 40% (tornando-a assim uma pessoa de interesse em potencial).
# 
# 
# - Uma vez efetuada a criação e alteração dos atributos desejados, foi feita uma alteração final nos atributos total_payments e total_stock_value: em função da variação entre os valores máximos e minimos serem elevadas, aplicou-se a transformação destes para os seus valores logaritmicos.

# In[18]:

### > Criação de Features Auxiliares

### Exibe dataset POI atualizado
print "> Dataset POI atualizado antes da inclusão e alteração de atributos:", len(df)
x = df.poi.value_counts()
print "- Número de pessoas envolvidas na fraude:", x[1]
print "- Número de pessoas não envolvidas na fraude:", x[0]

print ""
print " ---------------------------------------------"
print "> Inclusão de atributos adicionais ao dataset:"
print " ---------------------------------------------"

# calcula o percentual de mensagens que foram recebidas de um POI
print "- msgfrom_POI: percentual de mensagens que foram recebidas de um POI"
df['msgfromPOI_ratio'] = (df['from_poi_to_this_person']/df['to_messages'])*100
df['msgfromPOI_ratio'].fillna(0,inplace=True)
df['msgfromPOI_ratio'] = df['msgfromPOI_ratio'].apply(lambda x: round(x,2))

# calcula o percentual de mensagens que foram enviadas para um POI
print "- msgto_POI: percentual de mensagens que foram enviadas para um POI"
df['msgtoPOI_ratio'] = (df['from_this_person_to_poi']/df['from_messages'])*100
df['msgtoPOI_ratio'].fillna(0,inplace=True)
df['msgtoPOI_ratio'] = df['msgtoPOI_ratio'].apply(lambda x: round(x,2))

# calcula o percentual de mensagens que foram enviadas para um POI
print "- sharedwithPOI: percentual de mensagens que foram enviadas para um POI"
df['sharedwithPOI_ratio'] = (df['shared_receipt_with_poi']/df['to_messages'])*100
df['sharedwithPOI_ratio'].fillna(0,inplace=True)
df['sharedwithPOI_ratio'] = df['sharedwithPOI_ratio'].apply(lambda x: round(x,2))

# recalculado o campo total_payments desconsiderando o atributo deferred_income (dedução)
df['total_payments'] = (df['bonus'] + df['salary'] + df['deferral_payments'] + df['loan_advances'] + 
                            df['expenses'] + df['other'] + df['long_term_incentive'] + df['director_fees'])

df = df[df.total_payments != 0]

# calcula o percentual de participação do bonus + salary sobre o total de pagamentos (total_payments)
print "- bonussalary_ratio: percentual de participação do bonus + salary sobre o total de pagamentos (total_payments)"
df['bonussalary_ratio'] = ((df['bonus']+df['salary'])/df['total_payments'])*100
df['bonussalary_ratio'] = df['bonussalary_ratio'].apply(lambda x: round(x,2))

df = df.replace(-0, 0)
df.fillna(0, inplace=True)

print ""
print " ------------------------------------------------"
print "> Alteração dos atributos existentes aos dataset:"
print " ------------------------------------------------"

print "- total_payments: coluna recalculada descontando o valor da coluna deferred_income (adiantamento)"
print "- total_payments: selecionados apenas os registros com valor diferentes de zero"
print "- total_payments: convertido para o seu valor logaritmico"
print "- total_stock_value: convertido para o seu valor logaritmico"
print "- %: conversão para % de todos os demais atributos financeiros (os nomes dos atributos não foram alterados)"

df['bonus'] = (df['bonus']/df['total_payments'])*100
df['bonus'] = df['bonus'].apply(lambda x: round(x,2))

df['salary'] = (df['salary']/df['total_payments'])*100
df['salary'] = df['salary'].apply(lambda x: round(x,2))

df['deferred_income'] = (df['deferred_income']/df['total_payments'])*-1*100
df['deferred_income'] = df['deferred_income'].apply(lambda x: round(x,2))

df['deferral_payments'] = (df['deferral_payments']/df['total_payments'])*100
df['deferral_payments'] = df['deferral_payments'].apply(lambda x: round(x,2))

df['loan_advances'] = (df['loan_advances']/df['total_payments'])*100
df['loan_advances'] = df['loan_advances'].apply(lambda x: round(x,2))

df['expenses'] = (df['expenses']/df['total_payments'])*100
df['expenses'] = df['expenses'].apply(lambda x: round(x,2))

df['other'] = (df['other']/df['total_payments'])*100
df['other'] = df['other'].apply(lambda x: round(x,2))

df['long_term_incentive'] = (df['long_term_incentive']/df['total_payments'])*100
df['long_term_incentive'] = df['long_term_incentive'].apply(lambda x: round(x,2))

df['director_fees'] = (df['director_fees']/df['total_payments'])*100
df['director_fees'] = df['director_fees'].apply(lambda x: round(x,2))

df['total_payments'] = np.log(df['total_payments'])
df.loc[df['total_stock_value']==0, ['total_stock_value']] = 1
df['total_stock_value'] = np.log(df['total_stock_value'])

df = df.replace(-0, 0)
df.fillna(0, inplace=True)

# adiciona novas features à lista POI (tipo 'c' = custom)
poi_features = insertfeature("msgfromPOI_ratio", "c", "p", poi_features, False)
poi_features = insertfeature("msgtoPOI_ratio", "c", "p", poi_features, False)
poi_features = insertfeature("sharedwithPOI_ratio", "c", "p", poi_features, False)
poi_features = insertfeature("bonussalary_ratio", "c", "p", poi_features, False)

### Exibe dataset POI atualizado
print ""
print "> Dataset POI atualizado após a inclusão e alteração de atributos:", len(df)
x = df.poi.value_counts()
print "- Número de pessoas envolvidas na fraude:", x[1]
print "- Número de pessoas não envolvidas na fraude:", x[0]


# # Passo 6: Remoção de Atributos sem Relevância ou sem significado conhecido

# - Após realizadas as alterações aos atributos existentes e criados novos atributos, nesta nova etapa será feita uma re-análise dos atributos restantes, a fim de verificar se os mesmos ainda possuem alguma relevância para a análise dos dados. Neste processo, foram removidos os atributos que de alguma forma possuem um atributo equivalente de maior relevância ou que para o qual não se conhece seu significado, requerendo portanto, um conhecimento da regra de negócio anterior à geração dos dados.

# In[19]:

#remove features sem significado conhecido
print ""
print " -------------------------------------------------------------------"
print "> Remoção de features sem relevância e/ou sem significado conhecido:"
print " -------------------------------------------------------------------"

print "- from_poi_to_this_person: substituída pelo atributo msgfromPOI_ratio"
poi_features = removefeature("from_poi_to_this_person", poi_features)

print "- to_messages: substituída pelo atributo msgfromPOI_ratio"
poi_features = removefeature("to_messages", poi_features)

print "- from_this_person_to_poi: substituída pelo atributo msgtoPOI_ratio"
poi_features = removefeature("from_this_person_to_poi", poi_features)

print "- from_messages: substituída pelo atributo msgtoPOI_ratio"
poi_features = removefeature("from_messages", poi_features)

print "- email_address: atributo sem relevância para as análises"
poi_features = removefeature("email_address", poi_features)

print "- shared_receipt_with_poi: substituída pelo atributo sharedwithPOI_ratio"
poi_features = removefeature("shared_receipt_with_poi", poi_features)

print "- restricted_stock_deferred: a relação deste atributos com os demais atributos stock requer conhecimento de regra de negócio"
poi_features = removefeature("restricted_stock_deferred", poi_features)

print "- restricted_stock: a relação deste atributos com os demais atributos stock requer conhecimento de regra de negócio"
poi_features = removefeature("restricted_stock", poi_features)

print "- exercised_stock_options: a relação deste atributos com os demais atributos stock requer conhecimento de regra de negócio"
poi_features = removefeature("exercised_stock_options", poi_features)


# # Parte II - Análise Exploratória de Dados
# # Passo 7: Dataset atualizado para o conjunto de atributos de interesse

# - Neste ponto já temos os atributos relevantes selecionados e tratados, possibilitando agora a exploração dos dados: analisar sua distribuição, frequência, e verificar novas necessidades de transformação e eliminação de outliers. 
# 
# 
# - Neste momento, verifica-se a importância de apresentar o dataset atualizado após concluídas as etapas de tratamento dos dados.

# In[20]:

### Atualização final do dataset com as features selecionadas
df_poi = f_list2ds(df, False, poi_features)

### Exibe dataset POI atualizado
print "> Dataset POI atualizado após criação e remoção de atributos:", len(df_poi)
x = df_poi.poi.value_counts()
print "- Número de pessoas envolvidas na fraude:", x[1]
print "- Número de pessoas não envolvidas na fraude:", x[0]

df_poi


# # Passo 8: Análise gráfica dos atributos de interesse

# - Nesta etapa será realizada uma análise gráfica dos atributos de interesse selecionados. Opcionalmente, pode-se remover os ouliers situados fora dos limites determinados por um percentual máximo e um percentual mínimo de frequência.
# 
# 
# - Entretanto, verificou-se que a remoção dos ouliers não resultaram em diferenças positivas significativas nos algoritmos de aprendizagem, onde, portanto, optou-se por manter todos os dados existentes.
# 
# 
# - Nas analises gráficas, foram exibidos histogramas, destacando o comportamento e distribuição da variável para as pessoas de interesse (envolvidas na fraude) - POI e as pessoas não envolvidas na fraude. Além deste também foi exibido um boxplot a fim de identificar a quantidade de pontos situada fora dos quadrantes e avaliar a necessidade ou não de remoção dos ouliers.
# 
# 
# - Os comentários acerca de cada atributo foram exibidos logo abaixo do seu respectivo gráfico.

# In[21]:

### > Exploring

restauradatasetoriginal = True
removeroutliers = False
permaxoutlier = 90
perminoutlier = 10

# atualiza dataset para re-execução da rotina
if (restauradatasetoriginal):
    df_poi = f_list2ds(df, False, poi_features)

# guarda o número de registros inicial do dataset
numreg = len(df_poi)

if (removeroutliers):
    print "> Exibe histograma dos atributos do dataset POI e remove outliers:"
else:
    print "> Exibe histograma dos atributos do dataset POI:"

# Inserir aqui as observações/conclusões acerca de cada gráfico
zeraobsfeatures(poi_features)
insertobsfeature("salary", poi_features, 
                 "Em sua maioria observa-se que o atributo salary dos POI constitui menos de 20% do total do montante pago aos mesmos.")
insertobsfeature("deferral_payments", poi_features,
                 "Em sua maioria observa-se que o atributo salary dos POI constitui menos de 20% do total do montante pago aos mesmos.")
insertobsfeature("total_payments", poi_features,
                 "Após a transformação para logaritmo, é possível observar a distribuição normal da frequência em torno da média.")
insertobsfeature("loan_advances", poi_features,
                 "Atributo praticamente não possui variância significativa que possa apontar alguma tendência.")
insertobsfeature("bonus", poi_features,
                 "Atributo parece se comportar da mesma forma, tanto para os POI quanto para os não-POI")
insertobsfeature("deferred_income", poi_features,
                 "Observa-se que mais 70% das pessoas não apresentaram desconto de valor (deduções).")
insertobsfeature("total_stock_value", poi_features,
                 "Apesar de muitas pessoas possuírem um total_stock_value=0, observa-se principalmente que isto ocorre apenas para os não-POI")
insertobsfeature("expenses", poi_features,
                 "Atributo parece se comportar da mesma forma, tanto para os POI quanto para os não-POI")
insertobsfeature("other", poi_features,
                 "Atributo parece se comportar da mesma forma, tanto para os POI quanto para os não-POI")
insertobsfeature("long_term_incentive", poi_features,
                 "Atributo com mais de 50% com valores zeros para os não-POI e sem distinção explicita para os POI")
insertobsfeature("director_fees", poi_features,
                 "Atributo praticamente sem variância significativa.")
insertobsfeature("msgfromPOI_ratio", poi_features,
                 "Atributo parece se comportar da mesma forma, tanto para os POI quanto para os não-POI")
insertobsfeature("msgtoPOI_ratio", poi_features,
                 "Atributo onde mais de 70% dos não-POI nunca enviaram e-mail para um POI. Isso significa grandes chances de quem envia e-mail para um POI ser também um.")
insertobsfeature("sharedwithPOI_ratio", poi_features,
                 "Atributo parece se comportar da mesma forma, tanto para os POI quanto para os não-POI")
insertobsfeature("bonussalary_ratio", poi_features,
                 "Atributo parece se comportar da mesma forma, tanto para os POI quanto para os não-POI, apesar de 35% dos não-POI não possuírem bonus e salários.")

df_poi = plotviewfeatures(df_poi, poi_features, removeroutliers, permaxoutlier, perminoutlier)

x = df_poi.poi.value_counts()

print ""
if (removeroutliers):
    print "  -------------------------------------------"
    print "> Observações Gerais após remoção dos ouliers:"
    print "  -------------------------------------------"
    print "- Número de registros antes:", numreg
    print "- Número de registros após:", len(df_poi)
    print "- Número de registros removidos:", numreg-len(df_poi)
    print "- Número de pessoas envolvidas na fraude:", x[1]
    print "- Número de pessoas não envolvidas na fraude:", x[0]
else:
    print "  -----------"
    print "> Observações:"
    print "  -----------"
    print "- Número de pessoas envolvidas na fraude:", x[1]
    print "- Número de pessoas não envolvidas na fraude:", x[0]
    print "- Número total de de pessoas do dataset:", x[0] + x[1]


# # Parte III - Aplicação de Machine Learning

# ## Task 1: Select what features you'll use
# 
# Nesta tarefa os dados preparados no formato pandas foram convertidos para numpy, e a lista de features recriada segundo o formato exigido na especificação do projeto.
# 
# Desta forma, a preparação dos dados, incluindo as tarefas de remoção de ouliers, limpeza, transformações e criação de novas features, já encontram-se concluídas neste etapa.

# In[22]:

### ---------------------------------------
### Task 1: Select what features you'll use.
### ---------------------------------------

# Remove features que não serão mais necessárias após a exploração dos dados
poi_features = removefeature("email_address", poi_features)
poi_features = removefeature("isoutlier", poi_features)

# atualiza novamente o dataset com a lista de features atualizada
df_poi = f_list2ds(df, False, poi_features)

# gera a features_list no formato solicitado na especificação do projeto
features_list = templst2featurelist(poi_features, features_list)

# converte o dataset de pandas para dicionário numpy
newdatadict = df_poi.set_index('name').to_dict(orient="index")

### Store to my_dataset for easy export below.
my_dataset = newdatadict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# ## Task 2: Remoção de outliers
# 
# Após a limpeza e transformações dos campos, não verificou-se diferenças significativas nos resultados removendo os outliers para as frequências maiores que 90% e menores que 10%. Desta forma, optou-se por preservar os dados em sua inteade, visto que os resultados obtidos com os algoritmos foram satisfatórios.

# ## Task 3: Create new feature(s)
# 
# Em relação ao dataset original, estão sendo considerados nesta etapa os seguintes atributos adicionais criados:
# 
# - msgfrom_POI: percentual de mensagens que foram recebidas de um POI
# - msgto_POI: percentual de mensagens que foram enviadas para um POI
# - sharedwithPOI: percentual de mensagens que foram enviadas para um POI
# - bonussalary_ratio: percentual de participação do bonus + salary sobre o total de pagamentos (total_payments)
# 

# ## Task 4: Try a varity of classifiers
# 
# Nesta etapa, tratou-se de observar os resultados dos algoritmos com os dados preparados. Os resultados obtidos são apresentados abaixo:
# 
# Conclui-se portanto, que nesta fase já foi possível atingir os resultados desejados através da aplicação do algoritmo DecisionTree, pois o mesmo apresentou uma taxa de Precision e Recall superiores à 0.3.
# 

# In[23]:

### -----------------------------------
### Task 4: Try a varity of classifiers
### -----------------------------------

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

# Gaussian Naive Bayes Classifier
print "---------------------------------------------------------"
print "> 1. Testing dataset with Gaussian Naive Bayes Classifier:"
print "---------------------------------------------------------"
clf_gnb = GaussianNB()
test_classifier(clf_gnb, my_dataset, features_list)
print ""

# Decision Tree Classifier
print "--------------------------------------------------"
print "> 2. Testing dataset with Decision Tree Classifier:"
print "--------------------------------------------------"
clf_dt = DecisionTreeClassifier(random_state=42)
test_classifier(clf_dt, my_dataset, features_list)
print ""

# KNeighborsClassifier
print "---------------------------------------------"
print "> 3. Testing dataset with KNeighborsClassifier"
print "---------------------------------------------"
clf_neigh = KNeighborsClassifier(n_neighbors=3)
test_classifier(clf_neigh, my_dataset, features_list)
print ""


# ## Task 5: Tune your classifier to achieve better than .3 precision and recall
# 
# A fim de melhorarmos ainda mais os resultados de nossa árvore de decisões, foi necessário compreender as métricas solicitadas, uma vez que 2 métricas precisam ser atinjidas simultâneamente e os algoritmos tratam apenas 1. Assim, a fim de aplicar uma métrica que pudesse compreender ambos scores acima, foi adotada a métrica f1.
# 
# > A pontuação F1 pode ser interpretada como uma média ponderada da precisão e do recall, onde uma pontuação F1 atinge seu melhor valor em 1 e a pior pontuação em 0. A contribuição relativa de precisão e recall para a pontuação F1 é igual. A fórmula para a pontuação F1 é:
# 
# > ### F1 = 2 x (precisão x recall) / (precisão + recall)

# Outro ponto implementado visando a melhoria dos scores precision e recall (por meio da métrica f1) foi de aplicar um algoritmo para seleção dos melhores atributos, Select Kbest, pois espera-se que utilizando somente as features de fato relevantes será possível obter melhores resultados.
# 
# O número de features desejadas foi calculado juntamente com os parâmetros da árvore de decisão e não isoladamente. Desta forma, o número de features selecionadas é o melhor número de features para os melhores parâmetros da árvore de decisão.
# 
# Os resultados obtidos usando o conjunto de testes são apresentados a seguir:

# In[24]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score

# Recursos de acordo com o requerido pela versão instalada 18.1
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

# --------------------------------------
# Test-Data: utiliza features and labels
# --------------------------------------
# Cria os datasets de treinamento e testes
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

# -------------------------------------------------------
# Create a pipeline with feature selection and classifier
# -------------------------------------------------------

# Cria uma lista com a sequência de 1 até o número total de features
n_features = np.arange(1, len(features_list))

# Cria um pipeline para processar a seleção de variáveis com SelectKBest junto com a DecisionTree
tree_pipe = Pipeline([
        ('select_features', SelectKBest()),
        ('classify', DecisionTreeClassifier())
    ])

# Define os parâmetros da SelectKBest e da DecisionTree que serão otimizados
param_grid = dict(select_features__k = n_features,
                  classify__criterion = ['gini', 'entropy'],
                  classify__splitter = ['random', 'best'],
                  classify__min_samples_split = [2, 4, 6, 8, 10, 20],
                  classify__min_samples_leaf = [1, 2, 4, 6, 8, 10, 20],
                  classify__max_depth = [None, 5, 10, 15, 20],
                  classify__random_state = [ 42 ])

# Cria o score a ser utilizado na GridSearchCV por meio da função f1_score
scorer = make_scorer(f1_score)

# Define as regras de validação cruzada para serem executadas no GridSearchCV
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

# Define o número de jobs para ser processado em paralelo
njobs=2

print "> Start at:", (time.strftime('%d/%m/%Y %H:%M:%S'))

# Cria o classificador otimizado com o GridSearchCV
tree_clf = GridSearchCV(tree_pipe, param_grid = param_grid, scoring=scorer, cv=sss, n_jobs=njobs)
# from train_test_split
#tree_clf = tree_clf.fit(features_train, labels_train)
# from feature_format
tree_clf = tree_clf.fit(features, labels)


# Exibe os resultados do classificador otimizado criado
print ""
print "> best f1-score:", tree_clf.best_score_
print ""
print tree_clf.best_params_
print "- ", tree_clf.best_estimator_
print ""
print "> End at:", (time.strftime('%d/%m/%Y %H:%M:%S'))
print ""


# In[25]:

# Relaciona as features selecionadas e seu grau de relevância

kbest = tree_clf.best_estimator_.named_steps['select_features']

features_array = np.array(features_list)
features_array = np.delete(features_array, 0)
indices = np.argsort(kbest.scores_)[::-1]
k_features = kbest.get_support().sum()

finalfeatures = []
for i in range(k_features):
    finalfeatures.append(features_array[indices[i]])

finalfeatures = finalfeatures[::-1]
scores = kbest.scores_[indices[range(k_features)]][::-1]

print "> Features selecionadas e seu grau de importancia"
for i in range(k_features):
    print "- ", features_array[indices[i]], " : ", round(kbest.scores_[indices[i]],5)
    
plt.barh(range(k_features), scores, color="blue")
plt.yticks(np.arange(0.4, k_features), finalfeatures)
plt.title('Features selecionadas e seu grau de importancia')
plt.show()


# ## Conclusão:
# 
# Como não verificou-se melhorias significativas no score f1 por meio da aplicação dos melhores parâmetros encontrados (em relação aos resultados originais obtidos por meio dos parâmetros originais do classificador DecisionTree com todas as features), optou-se por exportar o classificador DecisionTree com seus parâmetros default.
# 
# Acredita-se que o motivo de ter-se obtido resultados melhores antes da afinação dos parâmetros do algoritmo selecionado (Decision Tree) seja em função de que, com exceção de total_payments e total_stock_value, todos os demais algoritmos foram transformados para seu percentual de representatividade. Desta forma, eliminou-se as diferenças naturais decorrentes do cargo hierárquico das pessoas analisadas (exemplo: o salário de um diretor é naturalmente superior ao salário de um supervisor).

# In[26]:

print "> Teste do classificador otimizado utilizando a função tester.py:"
print ""
print "> Start at:", (time.strftime('%d/%m/%Y %H:%M:%S'))
print ""
test_classifier(tree_clf.best_estimator_, my_dataset, features_list)
print ""
print "> End at:", (time.strftime('%d/%m/%Y %H:%M:%S'))
print ""


# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# 
# Como não foi possível melhorar os resultados do algoritmo aplicado (decision tree), foi exportado o classificador original, antes de aplicadas as rotinas de melhoria dos parâmetros.
# 
# > Resultados atingidos:
# 
# > --------------------------------------------------
# > Testing dataset with Decision Tree Classifier:
# > --------------------------------------------------
# >DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
# >            max_features=None, max_leaf_nodes=None,
# >            min_impurity_split=1e-07, min_samples_leaf=1,
# >            min_samples_split=2, min_weight_fraction_leaf=0.0,
# >            presort=False, random_state=None, splitter='best')
# 
# >   Accuracy: 0.80362	Precision: 0.35030	Recall: 0.32350	F1: 0.33637	F2: 0.32853
# >	Total predictions: 13000	True positives:  647	False positives: 1200	False negatives: 1353	True negatives: 9800

# In[27]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(tree_clf.best_estimator_, my_dataset, features_list)
dump_classifier_and_data(clf_dt, my_dataset, features_list)

