
# Udacity: Programa Nanodegree - Formação Analista de Dados
# Projeto: Usando Machine Learning para identificar fraude na Enron Corpus
## Autor: Ubirajara Theodoro Schier
## Data: Novembro/2017

## Fraude na Enron: Resumo

Origem: Wikipédia, a enciclopédia livre.
A Enron Corporation foi uma companhia de energia americana, localizada em Houston, Texas. Empregava cerca de 21 000 pessoas, tendo sido uma das empresas líderes no mundo em distribuição de energia (electricidade, gás natural) e comunicações, mas decretou falência. Seu faturamento atingia 101 bilhões de dólares em 2000, pouco antes do escândalo financeiro que ocasionou sua falência.

Alvo de diversas denúncias de fraudes contábeis e fiscais e com uma dívida de 13 bilhões de dólares, o grupo pediu concordata em dezembro de 2001 e arrastou consigo a Arthur Andersen, que fazia a sua auditoria. Na época, as investigações revelaram que a Enron havia manipulado seus balanços , com a ajuda de empresas e bancos, e escondera dívidas de 25 bilhões de dólares por dois anos consecutivos, tendo inflado artificialmente os seus lucros.

O governo dos Estados Unidos abriu dezenas de investigações criminais contra executivos da Enron e da Arthur Andersen. A Enron foi também processada pelas pessoas lesadas. De acordo com os investigadores, os executivos e contadores, assim como instituições financeiras e escritórios de advocacia, que à época trabalhavam para a companhia, foram, de alguma forma e em diferentes graus, responsáveis pelo colapso da empresa.

#### Fonte: https://pt.wikipedia.org/wiki/Enron

## 1. Objetivo geral do projeto:

O objetivo deste projeto é o de aplicar os métodos de Data Wrangling e EAD (análise exploratória de dados) e incorporar à este último os recursos de Machine Learning aprendidos. Com a utilização deste conjunto de ferramentas sobre uma base de dados da Enron fornecida, tentar-se-á contribuir na classificação e identificação das pessoas envolvidas no escândalo ("poi" - person of interest).

A etapa de Data Wrangling contribuirá na estruturação e organização do conjunto de dados de acordo com as necessidades identificadas ao longo de todo projeto. A etapa EAD, por meio da visualização dos dados, possibilitará conhecer melhor os atributos, suas características, e seus interrelacionamentos. Por último, por meio da utlização de uma biblioteca de aprendizado de máquina de código aberto para a linguagem de programação Python (sklearn), serão desenvolvidos recursos preditivos para identificação das poi.

> Utilização:
 
> 1- Coloque todo o arquivo no diretório raiz
  
> 2- execute o "poi_id.py" em Python ou "poi_id.ipynb" em Jupyter notebook.

## 2. Conjunto de Dados:

Como etapa de pré-processamento deste projeto, foi fornecido no formato de um dicionário, os dados combinados da base "Enron email and financial". Os atributos nos dados possuem basicamente três tipos: atributos financeiros, de email e rótulos POI (pessoa de interesse).

atributos financeiros: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (todos em dólares americanos (USD))

atributos de email: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (as unidades aqui são geralmente em número de emails; a exceção notável aqui é o atributo ‘email_address’, que é uma string)

rótulo POI: [‘poi’] (atributo objetivo lógico (booleano), representado como um inteiro).

Este conjunto de dados inicial fornecido possui as seguintes características:

- Número de registros original do dataset: 146
- Número de pessoas envolvidas na fraude: 18
- Número de pessoas não envolvidas na fraude: 128
- Número de atributos do dataset: 23

## 3. Preparação do Conjunto de Dados:

Para que seja possível obter bons resultados na aplicação dos algoritmos de Machine Learning, é necessário analisar os dados fornecidos verificar se os mesmos não necessitam dos seguintes tratamentos: Limpeza, Transformação, Alteração, Criação, Tratamento de registros inválidos e Remoção de Outliers.

- Limpeza: Ao longo de todo o projeto procurou-se identificar dados inconsistentes que, se processados, poderiam distorcer os resultados finais. Tais registros de dados, por sua vez, foram eliminados. Exemplo: Foi excluído o registro onde o nome da pessoa era "TOTAL", pois este referia-se à um registro totalizador do conjunto de dados.


- Transformação: Por vezes alguns atributos possuem grandezas que para uma melhor visualização gráfica, precisam ser redimensionados em uma outra escala. Neste projeto, verificamos a necessidade de transformar para escala logarítmica os atributos: total_payments e total_stock_value.


- Alteração: Com um conhecimento mais aprofundado dos dados, por muitas vezes é possível utilizar os dados existentes e alterá-los, de forma que, sua forma alterada, possa vir a reduzir distorções e melhorar os resultados desejados. Neste projeto, umas das principais alterações realizadas foi alterar os atributos financeiros não totalizadores (total_payments e total_stock_value) de valor monetário para percentual. Observou-se por exemplo que um funcionário não pode ser considerado uma "poi" apenas por ter um alto valor de bonus pago em comparação aos demais, pois esta diferença pode estar naturalmente associada ao cargo do mesmo. Entretanto, a participação do valor pago em bonus à um funcionário, em relação ao total pago ao mesmo (total_payments) pode ser muito melhor aproveitado como parâmetro para identificação de uma "poi" quando comparada com à de outros funcionários.


- Criação: Ao analisar os atributos originalmente disponíveis, por vezes percebe-se que os mesmos podem ser combinados entre os mesmos ou então também associados à dados externos extraídos e incorporados ao conjunto de dados. Neste projeto, após um conhecimento maior dos dados, verificou-se a oportunidade de criar atributos adicionais para gerar melhores resultados. Os atributos criados foram:

> msgfrom_POI: percentual de mensagens que foram recebidas de um POI

> msgto_POI: percentual de mensagens que foram enviadas para um POI

> sharedwithPOI: percentual de mensagens que foram enviadas para um POI

> bonussalary_ratio: percentual de participação do bonus + salary sobre o total de pagamentos (total_payments)

- Tratamento de registros inválidos: Uma dos requisitos mais importantes que irá selecionar qualitativamente os dados e atributos é a forma como serão tratados os registros inválidos, ou seja: o que fazer quando na ausência da informação. Isso se torna crucial, pois quando maior a quantidade de registros com atributos sem informação (mais conhecidos como "NaN") menor será o tamanho da amostra (quantidade de registros). Neste projeto identificou-se que os atributos financeiros não totalizadores foram extraídos de maneira inapropriada quando não localizados. Por exemplo: Se durante a extração de dados não foi encontrado nenhum registro de pagamento de bônus para um determinado funcionário, o mesmo foi salvo como "NaN" quando, na verdade, deveria ter sido salvo como 0 (zero). Esta conclusão foi possível únicamente por que observou-se que a soma dos atributos financeiros, considerando os valores NaN como zero, fechavam com o valor dos atributos totalizadores. Dessa forma, foi possível preservar 100% dos registros que apresentavam algum atributo igual à NaN.

> 'total_payments' = 'bonus' + 'salary' + 'deferral_payments' + 'loan_advances' + 'expenses' + 'other' +
>                    'long_term_incentive' + 'director_fees'

- Remoção de Outliers: Como último procedimento, com os dados já "limpos" (etapa de Limpeza concluída), foi aproveitado o processamento para visualização dos dados por meio de gráficos e tratados também os outliers, ou seja, valores discrepantes candidatos potenciais à estarem incorretos e distorcerem os resultados finais. Os ouliers foram classificados de acordo com a regra IQR (Interquartile Range Rule). Utilizando esta regra, aplicou-se o percentual para valores mínimos válidos acima de 10% das ocorrências e para valores máximos válidos acima de 90% das ocorrências. Entretanto, acredita-se que os dados já se encontravam consistentes, pois não observou-se diferenças significativas nos resultados excluindo os ouliers. Em função disso, preferiu-se preservar os dados e não excluir os ouliers.

## 4. Aplicação dos Algoritmos:

Para selecionar o melhor algoritmo de aprendizagem de máquina para os dados preparados, foi dado como pré-requisito a utilização da função "test_classifier" fornecida juntamente com a especificação do projeto. Aplicou-se essa função e foram testados os algoritmos abaixo com seus respectivos resultados. Os mesmos utilizam os dados ajustados e os parâmetros padrão de cada algoritmo.

>---------------------------------------------------------
> - Testing dataset with Gaussian Naive Bayes Classifier:
>---------------------------------------------------------
>	Accuracy: 0.43792	Precision: 0.17628	Recall: 0.72250	F1: 0.28342

>--------------------------------------------------
> - Testing dataset with Decision Tree Classifier:
>--------------------------------------------------
>	Accuracy: 0.80438	Precision: 0.35535	Recall: 0.33350	F1: 0.34408

>---------------------------------------------
> - Testing dataset with KNeighborsClassifier:
>---------------------------------------------
>   Accuracy: 0.83777	Precision: 0.28794	Recall: 0.03700	F1: 0.06557	F2: 0.04481

## 5. Avaliação do Desempenho:

Para avaliar o desempenho do algoritmo, a biblioteca sklearn oferece diversas métricas, entre elas, destacamos aquelas que são também utilizadas pela função "test_classifier" fornecida:

> - PRECISION: A precisão é a razão tp / (tp + fp) onde tp é o número de positivos verdadeiros e fp o número de falsos positivos. A precisão é intuitivamente a capacidade do classificador de não rotular como positivo uma amostra que é negativa (o melhor valor é 1 e o pior valor é 0).

Neste projeto, Precision nos indica a capacidade do algoritmo em classificar uma pessoa como "poi", sendo que esta é efetivamente uma "poi". Metaforicamente falando, quanto menor o número pessoas efetivamente "inocentes" ("não-poi") identificados como "culpadas" ("poi"), melhor será o desempenho do classificador (quanto menor fp - falso positivo -, mais próximo de 1 será o score Precision).

> - RECALL: O recall é a razão tp / (tp + fn) onde tp é o número de positivos verdadeiros e fn o número de falsos negativos. O recall é intuitivamente a capacidade do classificador de encontrar todas as amostras positivas (o melhor valor é 1 e o pior valor é 0).

Neste projeto, Recall nos indica a capacidade do algoritmo em classificar uma pessoa como "não-poi", sendo que esta é efetivamente uma "não-poi". Metaforicamente falando, quanto menor o número de pessoas efetivamente "culpadas" ("poi") identificadas como "inocentes" ("não-poi"), melhor será o desempenho do classificador (quanto menor fn - falso negativo -, mais próximo de 1 será o score Recall).

> - F1: A pontuação F1 pode ser interpretada como uma média ponderada da precisão e do recall, onde uma pontuação F1 atinge seu melhor valor em 1 e a pior pontuação em 0. A contribuição relativa de precisão e recall para a pontuação F1 é igual. A fórmula para a pontuação F1 é: F1 = 2 * (precisão * recall) / (precisão + recall)

Observa-se portanto, que o algoritmo DecisionTree apresentou o melhor desempenho utilizando todas as features ajustadas durante a fase de preparação de dados como, também, os parâmetros padrão deste algoritmo. Os requisitos do projeto pedem um desempenho mínimo de 0.3 para as métricas "Precision", e "Recall", o que foi alcançado nas condições mencionadas.

## 6. Otimização do algoritmo:

Neste próxima etapa, selecionaremos e otimizaremos os parâmetros do algoritmo que apresentou os melhores resultados para as métricas de Precision e Recall na etapa anterior, onde foram utilizados valores padrões dos parâmetros do algoritmo.

A otimização de um algoritmo classificador consiste em ajustar seus respectivos parâmetros em função das características do conjunto de dados que está sendo analisado, de forma a encontrar a melhor combinação de parâmetros que consiga extrair um desempenho ainda melhor do algoritmo (melhorando as métricas desejadas). Os modelos ou algoritmos de classificação podem ter muitos parâmetros e encontrar a melhor combinação de parâmetros pode ser tratado como um problema de busca.

Este processo pode ser manual, entretanto, serão utilizados recursos disponíveis que automatizam a "busca pela melhor combinação" dos parâmetros.

Um dos recursos implementados, a função Select KBest, permite obter melhoria dos resultados obtidos por meio da seleção das "n" features efetivamente relevantes, onde "n" será um dos parâmetros à ser buscado na otimização. A utilização das features efetivamente relevantes, proporcionam:

- Redução da superposição: menos dados redundantes significam menos oportunidade de tomar decisões com base no ruído.
- Melhora a precisão: menos dados enganosos significa que a precisão da modelagem melhora.
- Reduz o tempo de treinamento: Menos dados significam que os algoritmos treinam mais rápido.

O outro recurso implementado é propriamente a otimização de alguns dos principais parâmetros do algoritmo DecisionTree selecionado. Abaixo, pode-se observar os parâmetros selecionados para otimização, bem como os valores que serão "testados" em cada um deles (entre colchetes):
                  
                  Parâmetro:            Valores testados:
                  criterion           = ['gini', 'entropy']
                  splitter            = ['random', 'best']
                  min_samples_split   = [2, 4, 6, 8, 10, 20]
                  min_samples_leaf    = [1, 2, 4, 6, 8, 10, 20]
                  max_depth           = [None, 5, 10, 15, 20]
                  random_state        = [ 42 ] (utilizou-se apenas o valor 42 para compatibilizar com a função teste.py)

É importante destacar que quanto maior o número de parâmetros à serem otimizados, maior será o tempo de busca pela melhor combinação entre os mesmos.

Para otimizar os parâmetros de ambos recursos, foi utilizado um recurso chamado GridSearchCV, que possibilita "varrer" todos os parâmetros que se deseja melhorar até atingir o melhor resultado da métrica especificada. A métrica selecionada por sua vez foi a F1, pois por meio da melhoria desta métrica espera-se obter melhores resultados tanto para a métrica Precision, quanto para a métrica Recall.

Como resultado, obtemos um score F1 igual à 0.31402, utilizando os seguintes parâmetros:

> Número de features selecionadas: 11 (select_features_k: 11)

               > Select KBest - Features e seu grau de importância:
                 - total_payments       :  12.72112
                 - bonus                :  6.16437
                 - total_stock_value    :  5.24066
                 - loan_advances        :  4.84807
                 - director_fees        :  2.41499
                 - deferral_payments    :  2.12845
                 - long_term_incentive  :  0.9983
                 - expenses             :  0.92709 
                 - deferred_income      :  0.38889
                 - other                :  0.11358
                 - salary               :  1e-05

Observa-se acima que as novas features criadas não foram selecionadas, e, portanto, não tiveram influência no desempenho do classificador. Alguma delas, tais como: msgfrom_POI, msgto_POI e sharedwithPOI, poderiam por sua vez causar vazamento de dados, pois têm em seu conteúdo a informação que pretende-se prever (poi). Tais features devem ser evitadas, pois sutilmente e inadvertidamente podem resultar na sobre-estimativa do desempenho do algoritmo e superposição.

> DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
>             max_features=None, max_leaf_nodes=None,
>             min_impurity_split=1e-07, min_samples_leaf=1,
>             min_samples_split=6, min_weight_fraction_leaf=0.0,
>             presort=False, random_state=42, splitter='random'))])


## 7. Validação do Resultado Obtido:

Para validar os novos resultados obtidos por meio da afinação do algoritmo selecionado (DecisionTree), torna-se necessário submeter também à função "test_classifier", as features mais relevantes selecionadas e os melhores parâmetros obtidos, a fim de verificar se os resultados são compatíveis.

A validação do algoritmo implementado pela função "test_classifier", utiliza um recurso "cross-validation". Este recurso permite que os resultados obtidos não se limitem à somente à um conjunto de dados de testes. Desta forma, elimina-se a probabilidade de ocorrer uma "causalidade", onde a aplicação de um específico conjunto de dados de testes possam gerar resultados distorcidos. Para isso, o resultado da métrica obtido é calculado sobre a média dos resultados dos conjuntos de dados de testes verificados.

Neste projeto, como o conjunto de dados possui poucos registros e está desbalanceado (número de "não-poi" muito maior que o número de "poi"), optou-se pela utilização do algoritmo de validação cruzada "StratifiedShuffleSplit", pois este além de "embaralhar" os registros antes de separar os conjuntos de testes, também garante que em cada conjunto de teste existirá a mesma proporção entre "poi" e "não-poi". Com isso, anula-se a possibilidade da interferência da ordem de geração dos dados nos resultados finais, como também, a possibilidade do algoritmo processar conjunto de testes sem nenhum registro "poi".

Outros algoritmos como KFold apenas segmentam o conjunto de dados em blocos de testes, sem se preocupar com a questão de uma possível ordenação dos dados ou se haverá registros "poi" e "não-poi" nos blocos de testes gerados.

Os resultados dos testes do algoritmo otimizado (DecisionTree), foram:

> Resultados testes finais usando a DecisionTree com os parâmetros otimizados na função "test_classifier":

Accuracy: 0.75846	Precision: 0.20251	Recall: 0.19400	F1: 0.19816	F2: 0.19564
	Total predictions: 13000	True positives:  388	False positives: 1528	False negatives: 1612	True negatives: 9472

## 8. Conclusões Finais:



Abaixo o resumo dos resultados obtidos:

> Resultado-1: F1-score/test_classifier, com classificador DecisionTree não otimizado: > F1: 0.34408

> Resultado-2: F1-score/gridsearchcv, com classificador DecisionTree otimizado: > F1: 0.31402

> Resultado-3: F1-score/test_classifier, com classificador DecisionTree otimizado: > F1: 0.19816

#### Dificuldades Tecnicas encontradas:

Esperava-se que o Resultado-3 (classificador otimizado) fosse superior ao Resultado-1 (classificador não otimizado), já que os mesmos utilizaram a mesma função de avaliação de desempenho (test_classifier). Para entendermos o motivo do ocorrido, analisamos o código e constatamos algumas questões técnicas importantes à serem destacadas:

- a função "test_classifier" utilizou funções próprias para criação dos datasets de treinamento e testes (featureFormat e targetFeatureSplit) e não a função sugerida para implementação (train_test_split). Para compatibilizar esta diferença, foi utilizado os mesmos datasets utilizados pela função "test_classifier" (features and labels);


- a função "test_classifier" utiliza uma função chamada "StratifiedShuffleSplit" importada da biblioteca "sklearn.cross_validation". Entretanto, esta e outras funções da versão 0.18.1. do scikit-learn, são importadas da biblioteca "sklearn.model_selection". Entre elas, está função que otimiza os parâmetros do classificados "GridSearchCV", que na versão 0.18.1. do scikit-learn é importada SOMENTE da biblioteca "sklearn.model_selection". Tentou-se aplicar a mesma função "StratifiedShuffleSplit" utilizada no "test_classifier", no "GridSearchCV". Entretanto, o tempo de execução do algoritmo de otimização se mostrou inviável (5 horas de processamento), indicando que alguma coisa não estava certa. Por isso, optou-se por respeitar a versão instalada e utilizar todas as funções da biblioteca "sklearn.model_selection" no projeto.


Em função do segundo item acima exposto, a função que define o validador utilizado na função "GridSearchCV" não é o mesma que o utilizado pela função "test_classifier", em virtude da diferença entre as versões da biblioteca instalada e a versão da biblioteca utilizada pela versão de avaliação fornecida ("test_classifier"). Suspeita-se que tal fato possa vir a ser a explicação técnica do ocorrido.

#### Escolha do classificador:

Como o Resultado-1 e o Resultado-2 se mostraram compatíveis, optou-se por submeter o classificador não-otimizado, cujo resultado (Resultado-1), pois, além de já ter atingido o mínimo de 0.3 para os scores de Precision e Recall solicitados na especificação do projeto, também contorna-se as questões de incompatibilidade detectadas.

#### Considerações:

Importante destacar que os inúmero recursos computacionais disponíveis na biblioteca "scikit-learn" necessitam ser utilizados com bastante atenção, pois qualquer incompatibilidade entre os recursos pode vir a distorcer os resultados obtidos e comprometendo assim, os objetivos desejados. Novamente, a padronização se faz importante a fim de compatibilizar os requisitos utilizados nos diversos recursos disponíveis.
