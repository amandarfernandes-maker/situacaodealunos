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
    page_title="EduAI | Desempenho Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. ESTILO VISUAL
# ==========================================================

st.markdown("""
<style>

    /* Fundo geral */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #1e293b 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Título principal */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.90);
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15,23,42,0.06);
        transition: 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(15,23,42,0.10);
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Card de resultado */
    .result-card {
        padding: 28px;
        border-radius: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 10px 30px rgba(15,23,42,0.10);
    }

    .result-title {
        font-size: 15px;
        font-weight: 600;
        opacity: 0.75;
    }

    .result-value {
        font-size: 34px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Seções */
    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Botão */
    .stButton > button {
        border-radius: 12px;
        height: 48px;
        font-size: 16px;
        font-weight: 700;
        border: none;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #6366f1
        );
        color: white;
        box-shadow: 0 6px 18px rgba(79,70,229,0.25);
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #4338ca,
            #4f46e5
        );
        transform: translateY(-1px);
    }

    /* Inputs */
    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Rodapé */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 25px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. GERAÇÃO DA BASE DE DADOS
# ==========================================================

np.random.seed(42)

quantidade_alunos = 500

horas_estudo = np.random.randint(0, 16, quantidade_alunos)
faltas = np.random.randint(0, 31, quantidade_alunos)
nota = np.round(
    np.random.uniform(0, 10, quantidade_alunos),
    1
)


def classificar_aluno(horas, faltas, nota):

    if nota >= 7 and faltas <= 10:
        return "Aprovado"

    elif nota < 5 or faltas > 20:
        return "Reprovado"

    else:
        return "Recuperação"


situacao = [
    classificar_aluno(h, f, n)
    for h, f, n in zip(
        horas_estudo,
        faltas,
        nota
    )
]


df = pd.DataFrame({
    "Horas_de_estudo": horas_estudo,
    "Faltas": faltas,
    "Nota": nota,
    "Situacao": situacao
})


# ==========================================================
# 4. VARIÁVEIS
# ==========================================================

x = df[
    [
        "Horas_de_estudo",
        "Faltas",
        "Nota"
    ]
]

y = df["Situacao"]


# ==========================================================
# 5. TREINO E TESTE
# ==========================================================

x_train, x_teste, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# 6. MODELO
# ==========================================================

modelo = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

modelo.fit(x_train, y_train)


# ==========================================================
# 7. AVALIAÇÃO
# ==========================================================

previsoes_teste = modelo.predict(x_teste)

acuracia = accuracy_score(
    y_teste,
    previsoes_teste
)


# ==========================================================
# 8. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0 25px 0;">
            <div style="font-size:55px;">🎓</div>
            <h2 style="margin:0;">EduAI</h2>
            <p style="opacity:0.7;">
                Análise Inteligente
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📌 Sobre o sistema")

    st.write(
        """
        Este dashboard utiliza **Machine Learning**
        para analisar o desempenho escolar dos alunos.
        """
    )

    st.markdown("### 🤖 Algoritmo")

    st.info(
        "Árvore de Decisão\n\n"
        "Classificação baseada em:\n"
        "- Horas de estudo\n"
        "- Faltas\n"
        "- Nota"
    )

    st.divider()

    st.markdown("### 📊 Base de dados")

    st.metric(
        "Total de alunos",
        len(df)
    )

    st.metric(
        "Acurácia",
        f"{acuracia * 100:.1f}%"
    )


# ==========================================================
# 9. CABEÇALHO
# ==========================================================

st.markdown(
    '<div class="main-title">🎓 Análise de Desempenho Escolar</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Utilize inteligência artificial para analisar e prever
        a situação acadêmica dos alunos.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 10. CARDS SUPERIORES
# ==========================================================

aprovados = (df["Situacao"] == "Aprovado").sum()
recuperacao = (df["Situacao"] == "Recuperação").sum()
reprovados = (df["Situacao"] == "Reprovado").sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">👨‍🎓 Alunos analisados</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🟢 Aprovados</div>
            <div class="metric-value" style="color:#16a34a;">
                {aprovados}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🟡 Recuperação</div>
            <div class="metric-value" style="color:#d97706;">
                {recuperacao}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🔴 Reprovados</div>
            <div class="metric-value" style="color:#dc2626;">
                {reprovados}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ==========================================================
# 11. ÁREA DE ANÁLISE
# ==========================================================

st.markdown(
    '<div class="section-title">📝 Análise individual</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Informe os dados do aluno para que o modelo realize
        a classificação.
    </div>
    """,
    unsafe_allow_html=True
)


input_card = st.container()

with input_card:

    col1, col2, col3 = st.columns(3)

    with col1:

        horas = st.number_input(
            "📚 Horas de estudo",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

    with col2:

        faltas_aluno = st.number_input(
            "📅 Quantidade de faltas",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0
        )

    with col3:

        nota_aluno = st.number_input(
            "⭐ Nota final",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    st.markdown("<br>", unsafe_allow_html=True)

    analisar = st.button(
        "🔍  ANALISAR DESEMPENHO",
        type="primary",
        use_container_width=True
    )


# ==========================================================
# 12. RESULTADO
# ==========================================================

if analisar:

    novo_aluno = pd.DataFrame(
        [[
            horas,
            faltas_aluno,
            nota_aluno
        ]],
        columns=[
            "Horas_de_estudo",
            "Faltas",
            "Nota"
        ]
    )

    previsao = modelo.predict(
        novo_aluno
    )[0]

    probabilidades = modelo.predict_proba(
        novo_aluno
    )[0]

    indice = list(
        modelo.classes_
    ).index(previsao)

    confianca = probabilidades[indice] * 100


    st.divider()

    st.markdown(
        '<div class="section-title">📊 Resultado da análise</div>',
        unsafe_allow_html=True
    )


    # ------------------------------------------------------
    # Cor do resultado
    # ------------------------------------------------------

    if previsao == "Aprovado":

        cor = "#dcfce7"
        cor_texto = "#15803d"
        emoji = "🟢"
        mensagem = "Excelente! O aluno apresenta um bom desempenho."

    elif previsao == "Recuperação":

        cor = "#fef3c7"
        cor_texto = "#b45309"
        emoji = "🟡"
        mensagem = "Atenção! O aluno pode precisar de acompanhamento."

    else:

        cor = "#fee2e2"
        cor_texto = "#b91c1c"
        emoji = "🔴"
        mensagem = "O aluno apresenta indicadores de baixo desempenho."


    # ------------------------------------------------------
    # Card principal
    # ------------------------------------------------------

    st.markdown(
        f"""
        <div class="result-card"
             style="background:{cor}; color:{cor_texto};">

            <div class="result-title">
                RESULTADO DA INTELIGÊNCIA ARTIFICIAL
            </div>

            <div class="result-value">
                {emoji} {previsao}
            </div>

            <div style="margin-top:10px;">
                {mensagem}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ------------------------------------------------------
    # Métricas
    # ------------------------------------------------------

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "⭐ Nota",
            f"{nota_aluno:.1f}"
        )

    with r2:
        st.metric(
            "📚 Horas de estudo",
            f"{horas:.1f} h"
        )

    with r3:
        st.metric(
            "📅 Faltas",
            f"{faltas_aluno:.0f}"
        )

    with r4:
        st.metric(
            "🤖 Confiança",
            f"{confianca:.1f}%"
        )


    # ------------------------------------------------------
    # Probabilidades
    # ------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📈 Probabilidade por situação</div>',
        unsafe_allow_html=True
    )

    prob_df = pd.DataFrame({
        "Situação": modelo.classes_,
        "Probabilidade": probabilidades * 100
    })

    col_grafico, col_info = st.columns(
        [2, 1]
    )

    with col_grafico:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        cores = []

        for classe in modelo.classes_:

            if classe == "Aprovado":
                cores.append("#22c55e")

            elif classe == "Recuperação":
                cores.append("#f59e0b")

            else:
                cores.append("#ef4444")


        barras = ax.barh(
            prob_df["Situação"],
            prob_df["Probabilidade"],
            color=cores,
            height=0.55
        )

        ax.set_xlim(0, 100)

        ax.set_xlabel(
            "Probabilidade (%)"
        )

        ax.spines[
            ["top", "right", "left"]
        ].set_visible(False)

        ax.grid(
            axis="x",
            alpha=0.2
        )

        ax.set_axisbelow(True)

        for barra, valor in zip(
            barras,
            prob_df["Probabilidade"]
        ):

            ax.text(
                valor + 1,
                barra.get_y()
                + barra.get_height() / 2,
                f"{valor:.1f}%",
                va="center",
                fontweight="bold"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col_info:

        st.markdown(
            """
            ### 💡 Interpretação

            O modelo analisa simultaneamente:

            **📚 Horas de estudo**  
            Quanto tempo o aluno dedica aos estudos.

            **📅 Faltas**  
            Frequência escolar do aluno.

            **⭐ Nota**  
            Resultado acadêmico obtido.

            A previsão representa a classe
            com maior probabilidade segundo
            a Árvore de Decisão.
            """
        )


# ==========================================================
# 13. MODELO DE MACHINE LEARNING
# ==========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Modelo de Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Explore como o modelo foi treinado e avalie seu desempenho.
    </div>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs([
    "🌳 Árvore de Decisão",
    "🎯 Matriz de Confusão",
    "📋 Relatório de Classificação"
])


# ==========================================================
# 14. ÁRVORE
# ==========================================================

with tab1:

    st.info(
        "A árvore mostra as regras utilizadas pelo modelo "
        "para classificar os alunos."
    )

    fig, ax = plt.subplots(
        figsize=(22, 11)
    )

    plot_tree(
        modelo,
        feature_names=x.columns,
        class_names=modelo.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    ax.set_title(
        "Árvore de Decisão — Desempenho Escolar",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ==========================================================
# 15. MATRIZ DE CONFUSÃO
# ==========================================================

with tab2:

    matriz = confusion_matrix(
        y_teste,
        previsoes_teste,
        labels=modelo.classes_
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    imagem = ax.imshow(
        matriz,
        cmap="Blues"
    )

    ax.set_title(
        "Matriz de Confusão",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Previsão do Modelo"
    )

    ax.set_ylabel(
        "Valor Real"
    )

    ax.set_xticks(
        range(len(modelo.classes_))
    )

    ax.set_yticks(
        range(len(modelo.classes_))
    )

    ax.set_xticklabels(
        modelo.classes_
    )

    ax.set_yticklabels(
        modelo.classes_
    )


    for i in range(
        len(modelo.classes_)
    ):

        for j in range(
            len(modelo.classes_)
        ):

            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white"
                if matriz[i, j] >
                matriz.max() * 0.5
                else "#1e293b"
            )


    plt.colorbar(
        imagem,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)


# ==========================================================
# 16. RELATÓRIO
# ==========================================================

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">
                    🎯 ACURÁCIA DO MODELO
                </div>

                <div class="metric-value"
                     style="color:#4f46e5;">
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
                {acuracia * 100:.2f}%
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-title">
                    🧪 AMOSTRAS DE TESTE
                </div>

                <div class="metric-value">
                    {len(x_teste)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    relatorio = classification_report(
        y_teste,
        previsoes_teste,
        output_dict=True
    )

    relatorio_df = pd.DataFrame(
        relatorio
    ).transpose()

    relatorio_df = relatorio_df.round(3)

    st.dataframe(
        relatorio_df,
        use_container_width=True,
        height=300
    )


# ==========================================================
# 17. BASE DE DADOS
# ==========================================================

st.divider()

st.markdown(
    '<div class="section-title">🗃️ Base de dados</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Visualização dos dados utilizados para treinar o modelo.
    </div>
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "👨‍🎓 Total de alunos",
        len(df)
    )

with col2:
    st.metric(
        "🧠 Variáveis",
        3
    )

with col3:
    st.metric(
        "🎯 Acurácia",
        f"{acuracia * 100:.2f}%"
    )


with st.expander(
    "👀 Visualizar primeiros registros"
):

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 18. RODAPÉ
# ==========================================================

st.markdown(
    """
    <div class="footer">
        🎓 EduAI • Análise de Desempenho Escolar<br>
        Projeto demonstrativo de Machine Learning com Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
