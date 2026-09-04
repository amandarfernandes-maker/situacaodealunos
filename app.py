import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Análise de Desempenho Escolar",
    page_icon="📚",
    layout="wide"
)


# ==========================================================
# 2. GERANDO UMA BASE DE DADOS MAIOR
# ==========================================================

np.random.seed(42)

quantidade_alunos = 500

horas_estudo = np.random.randint(0, 16, quantidade_alunos)
faltas = np.random.randint(0, 31, quantidade_alunos)
nota = np.round(np.random.uniform(0, 10, quantidade_alunos), 1)


def classificar_aluno(horas, faltas, nota):
    # Aprovado:
    # Nota >= 7 e poucas faltas
    if nota >= 7 and faltas <= 10:
        return "Aprovado"

    # Reprovado:
    # Nota muito baixa ou muitas faltas
    elif nota < 5 or faltas > 20:
        return "Reprovado"

    # Recuperação:
    # Situação intermediária
    else:
        return "Recuperação"


situacao = [
    classificar_aluno(h, f, n)
    for h, f, n in zip(horas_estudo, faltas, nota)
]


df = pd.DataFrame({
    "Horas_de_estudo": horas_estudo,
    "Faltas": faltas,
    "Nota": nota,
    "Situacao": situacao
})


# ==========================================================
# 3. SEPARANDO VARIÁVEIS
# ==========================================================

x = df[["Horas_de_estudo", "Faltas", "Nota"]]
y = df["Situacao"]


# ==========================================================
# 4. TREINO E TESTE
# ==========================================================

x_train, x_teste, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# 5. CRIANDO O MODELO
# ==========================================================

modelo = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

modelo.fit(x_train, y_train)


# ==========================================================
# 6. AVALIANDO O MODELO
# ==========================================================

previsoes_teste = modelo.predict(x_teste)
acuracia = accuracy_score(y_teste, previsoes_teste)


# ==========================================================
# 7. CABEÇALHO
# ==========================================================

st.title("📚 Análise de Desempenho Escolar")
st.caption("Preencha os dados do aluno para realizar a previsão usando uma Árvore de Decisão.")


# ==========================================================
# 8. FORMULÁRIO / ENTRADAS
# ==========================================================

st.subheader("📝 Dados do aluno")

col1, col2, col3 = st.columns(3)

with col1:
    horas = st.number_input(
        "Horas de estudo",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

with col2:
    faltas_aluno = st.number_input(
        "Quantidade de faltas",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )

with col3:
    nota_aluno = st.number_input(
        "Nota",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


# ==========================================================
# 9. BOTÃO DE PREVISÃO
# ==========================================================

if st.button("🔍 Analisar Aluno", type="primary", use_container_width=True):

    novo_aluno = pd.DataFrame(
        [[horas, faltas_aluno, nota_aluno]],
        columns=["Horas_de_estudo", "Faltas", "Nota"]
    )

    previsao = modelo.predict(novo_aluno)[0]

    probabilidades = modelo.predict_proba(novo_aluno)[0]
    indice = list(modelo.classes_).index(previsao)
    confianca = probabilidades[indice] * 100

    st.divider()
    st.subheader("📊 Resultado da análise")

    if previsao == "Aprovado":
        st.success(f"### Situação: {previsao}")
    elif previsao == "Recuperação":
        st.warning(f"### Situação: {previsao}")
    else:
        st.error(f"### Situação: {previsao}")

    st.metric("Confiança aproximada", f"{confianca:.1f}%")

    st.info(
        f"**Detalhes do aluno**\n\n"
        f"- Horas de estudo: {horas:.1f}\n"
        f"- Faltas: {faltas_aluno:.0f}\n"
        f"- Nota: {nota_aluno:.1f}"
    )

    # Probabilidades por classe
    st.subheader("Probabilidades por situação")

    prob_df = pd.DataFrame({
        "Situação": modelo.classes_,
        "Probabilidade": probabilidades * 100
    })

    st.bar_chart(
        prob_df.set_index("Situação")["Probabilidade"]
    )


# ==========================================================
# 10. VISUALIZAÇÕES DO MODELO
# ==========================================================

st.divider()
st.subheader("🤖 Modelo de Machine Learning")

tab1, tab2, tab3 = st.tabs([
    "🌳 Árvore de Decisão",
    "📊 Matriz de Confusão",
    "📋 Relatório"
])


with tab1:
    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(
        modelo,
        feature_names=x.columns,
        class_names=modelo.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    ax.set_title("Árvore de Decisão - Desempenho dos Alunos")
    st.pyplot(fig)
    plt.close(fig)


with tab2:
    matriz = confusion_matrix(
        y_teste,
        previsoes_teste,
        labels=modelo.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    imagem = ax.imshow(matriz)

    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Valor Real")

    ax.set_xticks(range(len(modelo.classes_)))
    ax.set_yticks(range(len(modelo.classes_)))

    ax.set_xticklabels(modelo.classes_)
    ax.set_yticklabels(modelo.classes_)

    for i in range(len(modelo.classes_)):
        for j in range(len(modelo.classes_)):
            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar(imagem, ax=ax)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


with tab3:
    st.metric(
        "Acurácia no teste",
        f"{acuracia * 100:.2f}%"
    )

    relatorio = classification_report(
        y_teste,
        previsoes_teste,
        output_dict=True
    )

    relatorio_df = pd.DataFrame(relatorio).transpose()

    st.dataframe(
        relatorio_df.round(3),
        use_container_width=True
    )


# ==========================================================
# 11. INFORMAÇÕES DA BASE
# ==========================================================

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Modelo treinado com:** {len(df)} alunos")

with col2:
    st.write(f"**Acurácia no teste:** {acuracia * 100:.2f}%")

with st.expander("👀 Visualizar primeiros registros da base"):
    st.dataframe(df.head(20), use_container_width=True)
