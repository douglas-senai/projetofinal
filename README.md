# Análise de Risco de Crédito - Porjeto Senai [M1]

Descrição:
Este projeto visa desenvolver e avaliar modelos de Machine Learning para prever o risco de inadimplência em solicitações de empréstimo. Para instituições financeiras, a capacidade de identificar clientes com alta probabilidade de não cumprir suas obrigações financeiras é crucial para mitigar perdas, otimizar a concessão de crédito e manter a saúde financeira.

O objetivo principal é construir um modelo que possa classificar com precisão se um solicitante de empréstimo irá pagar (classe 0) ou se tornar inadimplente (classe 1), priorizando a minimização de Falsos Negativos (conceder empréstimo a um mau pagador) devido ao seu alto impacto financeiro direto.

---

## Guia de Execução

### Pré-requisitos
* Ter o Python 3.x instalado em sua máquina.
* Ter as seguintes bibliotecas:
* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* import seaborn as sns
* from sklearn.model_selection import train_test_split
* from imblearn.over_sampling import SMOTE
* from sklearn.preprocessing import StandardScaler
* from sklearn.neighbors import KNeighborsClassifier
* from sklearn.metrics import accuracy_score
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.metrics import classification_report, confusion_matrix

### Passo a Passo

1. **Clone este repositório** para a sua máquina local utilizando o terminal ou prompt de comando:
   ```bash
   git clone https://github.com/douglas-senai/projetofinal.git
   ```

2. **Navegue até o diretório** do projeto:
   ```bash
   cd projetofinal
   ```
3. **Certifique-se de que os arquivos de dados** (`credit_risk_dataset.csv`) estão localizados na mesma pasta do seu arquivo do notebook ou script Python.

4. **Execute o arquivo:**
   * Utilize o arquivo `.ipynb`, abra-o através do **Jupyter Notebook**, **VS Code** ou **Google Colab** e execute todas as células (`Run All`).

#### Resumo Executivo
**Principais Insights da Análise Exploratória de Dados (EDA)
Valores Nulos: Foram identificados e tratados valores nulos nas colunas person_emp_length e loan_int_rate através da imputação pela mediana para evitar perda de dados e vieses.
Outliers: Observou-se a presença de outliers em person_age (idades muito altas) e person_emp_length (tempos de emprego excessivamente longos), que foram ajustados para a mediana, considerando-os como dados irrealistas.
Balanceamento de Classes: A variável alvo loan_status (inadimplência) apresentou um desbalanceamento significativo, com a classe 'não inadimplente' sendo majoritária. Este problema foi mitigado utilizando a técnica SMOTE (Synthetic Minority Over-sampling Technique) no conjunto de treino, garantindo um melhor treinamento do modelo para ambas as classes.
Engenharia de Features: A coluna comprometimento_renda foi criada para representar o percentual do valor do empréstimo em relação à renda do solicitante, adicionando uma feature com alto poder preditivo.
Correlações: A análise de correlação revelou relacionamentos importantes entre as variáveis, que foram consideradas durante a seleção e pré-processamento das features.
Pré-processamento: Variáveis categóricas foram convertidas usando One-Hot Encoding, e variáveis numéricas contínuas foram escalonadas com StandardScaler para otimizar o desempenho dos modelos.
Veredito do Melhor Modelo
Foram avaliados dois modelos de classificação: K-Nearest Neighbors (KNN) e Árvore de Decisão.

KNN: Após otimização, o melhor modelo KNN (K=9) obteve uma acurácia de teste de 0.8548. O relatório de classificação mostrou um recall para a classe 1 (inadimplente) de 0.72.
Árvore de Decisão: Após otimização, o melhor modelo de Árvore de Decisão (max_depth=11) obteve uma acurácia de teste de 0.9122. O relatório de classificação para a classe 1 (inadimplente) apresentou um recall de 0.71.
Análise e Recomendação: No contexto de risco de crédito, a minimização de Falsos Negativos (FN) — prever que um cliente pagará, mas ele de fato se torna inadimplente — é geralmente mais crítica do que a minimização de Falsos Positivos (recusar um bom pagador). Um FN resulta em perda direta de capital para o banco.

Embora ambos os modelos tenham demonstrado boa performance, a Árvore de Decisão se destacou com uma acurácia geral ligeiramente superior e, mais importante, oferece alta interpretabilidade. A capacidade de entender e explicar as decisões do modelo é um fator chave para equipes de negócio em instituições financeiras, que precisam justificar as concessões ou recusas de crédito. O recall para a classe de inadimplência (0.71) é razoável, e a interpretabilidade compensa a pequena diferença em métricas em comparação com o KNN.

Portanto, a Árvore de Decisão é o modelo recomendado para implementação, dado seu bom equilíbrio entre performance preditiva (especialmente o recall para a classe de interesse) e interpretabilidade para as partes interessadas do negócio.
