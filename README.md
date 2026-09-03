📚 Sistema de Análise de Desempenho Escolar

Sistema desenvolvido em Python utilizando Machine Learning para prever a situação de um aluno com base em:

📖 Horas de estudo
❌ Quantidade de faltas
📝 Nota

O projeto utiliza uma Árvore de Decisão (DecisionTreeClassifier) para classificar os alunos em:

✅ Aprovado
⚠️ Recuperação
❌ Reprovado

Além da previsão, o sistema possui uma interface gráfica para entrada dos dados, visualização da árvore de decisão e matriz de confusão.

🚀 Funcionalidades
Geração de uma base de dados com 500 alunos.
Treinamento de um modelo de Machine Learning.
Classificação dos alunos em três situações.
Interface gráfica utilizando Tkinter.
Validação dos dados informados pelo usuário.
Exibição da previsão.
Exibição da confiança aproximada da previsão.
Cálculo da acurácia do modelo.
Relatório de classificação.
Visualização da Árvore de Decisão.
Visualização da Matriz de Confusão.
Botão para limpar os dados informados.
🧠 Como funciona

O sistema utiliza três características para realizar a previsão:

Horas de estudo
        +
Quantidade de faltas
        +
Nota
        ↓
Árvore de Decisão
        ↓
Situação do aluno


O modelo foi configurado com parâmetros para reduzir o risco de overfitting:

modelo = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

Parâmetros
Parâmetro	Valor	Função
max_depth	5	Limita a profundidade da árvore
min_samples_split	10	Mínimo de amostras para dividir um nó
min_samples_leaf	5	Mínimo de amostras em uma folha
random_state	42	Garante resultados reproduzíveis
📊 Base de dados

Para fins educacionais, o sistema gera automaticamente 500 registros sintéticos.

Os dados são criados utilizando valores aleatórios:

horas_estudo = np.random.randint(0, 16, quantidade_alunos)

faltas = np.random.randint(0, 31, quantidade_alunos)

nota = np.round(
    np.random.uniform(0, 10, quantidade_alunos),
    1
)


A situação do aluno é definida por regras:

✅ Aprovado

O aluno é considerado aprovado quando:

Nota >= 7
E
Faltas <= 10

❌ Reprovado

O aluno é considerado reprovado quando:

Nota < 5
OU
Faltas > 20

⚠️ Recuperação

Os demais casos são classificados como:

Recuperação


Importante: os dados utilizados no projeto são sintéticos. Portanto, a acurácia obtida serve principalmente para fins didáticos e não deve ser interpretada como desempenho de um modelo aplicado a dados escolares reais.

🏗️ Estrutura do projeto

Uma estrutura recomendada é:

analise-alunos/
│
├── main.py
├── README.md
└── requirements.txt


Onde:

main.py → código principal do sistema.
README.md → documentação do projeto.
requirements.txt → bibliotecas necessárias para execução.
💻 Tecnologias utilizadas
Python

Linguagem principal utilizada no desenvolvimento.

Pandas

Utilizado para criação e manipulação dos dados.

import pandas as pd

NumPy

Utilizado para geração dos dados sintéticos.

import numpy as np

Scikit-learn

Utilizado para treinamento, previsão e avaliação do modelo de Machine Learning.

from sklearn.tree import DecisionTreeClassifier


Também são utilizados:

train_test_split
accuracy_score
confusion_matrix
classification_report

Matplotlib

Utilizado para visualizar:

Árvore de Decisão
Matriz de Confusão
Tkinter

Utilizado para criar a interface gráfica do sistema.

📦 Instalação
1. Verifique se o Python está instalado

No terminal:

python --version


ou:

python3 --version


Recomenda-se utilizar uma versão recente do Python.

2. Clone o projeto
git clone https://github.com/seu-usuario/seu-repositorio.git


Entre na pasta:

cd analise-alunos

3. Crie um ambiente virtual

Windows:

python -m venv venv


Linux/macOS:

python3 -m venv venv

4. Ative o ambiente virtual
Windows
venv\Scripts\activate

Linux/macOS
source venv/bin/activate

5. Instale as dependências

Crie um arquivo chamado:

requirements.txt


Com:

pandas
numpy
scikit-learn
matplotlib


Depois execute:

pip install -r requirements.txt


O Tkinter normalmente já vem incluído nas instalações do Python para Windows e macOS. Em algumas distribuições Linux pode ser necessário instalá-lo separadamente.

▶️ Executando o projeto

Após instalar as dependências:

python main.py


A interface gráfica será aberta.

🖥️ Interface

A aplicação apresenta uma tela onde o usuário pode informar os dados de um aluno:

Horas de estudo:       [ 6  ]

Quantidade de faltas: [ 2  ]

Nota:                  [ 7.0 ]


Depois basta clicar em:

🔍 Analisar Aluno


O sistema realizará a previsão.

Exemplo:

Situação: Aprovado

Confiança aproximada: 95.0%

🌳 Árvore de Decisão

O botão:

🌳 Ver Árvore


abre uma representação gráfica da árvore utilizada pelo modelo.

Ela permite visualizar quais características foram utilizadas para tomar as decisões.

Exemplo conceitual:

             Nota <= 5.0?
              /        \
            SIM        NÃO
             |          |
        Reprovado    Faltas <= 10?
                       /       \
                     SIM       NÃO
                      |         |
                  Aprovado  Recuperação


A árvore real é construída automaticamente pelo algoritmo.

📊 Matriz de Confusão

O botão:

📊 Matriz


exibe a matriz de confusão do conjunto de teste.

Ela permite comparar:

          Previsão
        Aprovado
        Recuperação
        Reprovado


com os valores reais.

Isso ajuda a identificar quais classes o modelo está confundindo.

📈 Avaliação do modelo

O projeto separa os dados em:

80% → Treinamento
20% → Teste


Utilizando:

train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


A acurácia é calculada através de:

accuracy_score(
    y_teste,
    previsoes_teste
)


Também é gerado um relatório com:

classification_report(
    y_teste,
    previsoes_teste
)


Esse relatório apresenta métricas como:

Precision
Recall
F1-score
Support
🔮 Fazendo uma previsão

O sistema recebe os dados:

novo_aluno = pd.DataFrame(
    [[6, 2, 7.0]],
    columns=[
        "Horas_de_estudo",
        "Faltas",
        "Nota"
    ]
)


E realiza a previsão:

previsao = modelo.predict(novo_aluno)


Também é possível obter as probabilidades:

probabilidades = modelo.predict_proba(novo_aluno)

🛡️ Validação dos dados

A interface verifica se os valores informados são válidos.

Horas de estudo

Permitido:

0 até 24 horas

Faltas

Permitido:

0 até 100 faltas

Nota

Permitido:

0 até 10


Caso o usuário informe um valor inválido, o sistema apresenta uma mensagem de erro.

🎯 Objetivo do projeto

O objetivo principal é demonstrar, de forma prática, como utilizar Machine Learning para classificação e integrar um modelo de Inteligência Artificial a uma interface gráfica.

O projeto também serve como exemplo para estudar:

Python
Pandas
NumPy
Machine Learning
Classificação
Árvore de Decisão
Treinamento e teste
Avaliação de modelos
Interface gráfica
Visualização de dados
⚠️ Limitações

Este projeto possui caráter educacional.

A base utilizada é gerada artificialmente e as classificações são baseadas em regras previamente definidas.

Por isso:

A acurácia não representa necessariamente um cenário real.
O modelo não deve ser utilizado para decisões escolares reais sem validação.
Uma base real e suficientemente grande seria necessária para um sistema de produção.
Os critérios de aprovação, recuperação e reprovação podem variar de acordo com cada instituição.
🔮 Melhorias futuras

Algumas evoluções possíveis:

 Utilizar dados reais de alunos.
 Permitir importar arquivos CSV.
 Adicionar mais características ao modelo.
 Comparar Árvore de Decisão com Random Forest.
 Testar Logistic Regression.
 Testar K-Nearest Neighbors.
 Criar gráficos de desempenho.
 Criar histórico das previsões.
 Salvar os resultados em banco de dados.
 Criar sistema de login.
 Transformar a aplicação em sistema web.
 Criar API para realizar previsões.
 Implementar validação cruzada.
 Ajustar automaticamente os hiperparâmetros.
 Utilizar dados reais para avaliar o modelo.
📄 Licença

Este projeto pode ser utilizado para fins educacionais e de estudo.

👨‍💻 Autor

Projeto desenvolvido para estudos de Python, Machine Learning e Inteligência Artificial.

⭐ Se este projeto foi útil para seus estudos, considere deixar uma estrela no repositório!
