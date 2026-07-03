# Análise de Risco de Crédito - Porjeto Senai [M1]

Descrição:
Este projeto visa desenvolver e avaliar modelos de Machine Learning para prever o risco de inadimplência em solicitações de empréstimo. Para instituições financeiras, a capacidade de identificar clientes com alta probabilidade de não cumprir suas obrigações financeiras é crucial para mitigar perdas, otimizar a concessão de crédito e manter a saúde financeira.

O objetivo principal é construir um modelo que possa classificar com precisão se um solicitante de empréstimo irá pagar (classe 0) ou se tornar inadimplente (classe 1), priorizando a minimização de Falsos Negativos (conceder empréstimo a um mau pagador) devido ao seu alto impacto financeiro direto.

---

## Guia de Execução

### Pré-requisitos
* Ter o Python 3.x instalado em sua máquina.
* Ter as seguintes bibliotecas:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

### Passo a Passo

1. **Clone este repositório** para a sua máquina local utilizando o terminal ou prompt de comando:
   ```bash
   git clone https://github.com/douglas-senai/
   ```

2. **Navegue até o diretório** do projeto:
   ```bash
   cd 
   ```
3. **Certifique-se de que os arquivos de dados** (`credit_risk_dataset.csv`) estão localizados na mesma pasta do seu arquivo do notebook ou script Python.

4. **Execute o arquivo:**
   * Utilize o arquivo `.ipynb`, abra-o através do **Jupyter Notebook**, **VS Code** ou **Google Colab** e execute todas as células (`Run All`).
