import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


# ==========================================================
# 1. GERANDO UMA BASE DE DADOS MAIOR
# ==========================================================

np.random.seed(42)

quantidade_alunos = 500

horas_estudo = np.random.randint(0, 16, quantidade_alunos)
faltas = np.random.randint(0, 31, quantidade_alunos)
nota = np.round(np.random.uniform(0, 10, quantidade_alunos), 1)


# Regra para criar uma situação mais realista
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


# Criando DataFrame
df = pd.DataFrame({
    "Horas_de_estudo": horas_estudo,
    "Faltas": faltas,
    "Nota": nota,
    "Situacao": situacao
})


print("Primeiros registros:")
print(df.head())

print("\nQuantidade por situação:")
print(df["Situacao"].value_counts())


# ==========================================================
# 2. SEPARANDO VARIÁVEIS
# ==========================================================

x = df[["Horas_de_estudo", "Faltas", "Nota"]]
y = df["Situacao"]


# ==========================================================
# 3. TREINO E TESTE
# ==========================================================

x_train, x_teste, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# 4. CRIANDO O MODELO
# ==========================================================

modelo = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

modelo.fit(x_train, y_train)


# ==========================================================
# 5. AVALIANDO O MODELO
# ==========================================================

previsoes_teste = modelo.predict(x_teste)

acuracia = accuracy_score(y_teste, previsoes_teste)

print("\n==============================")
print("AVALIAÇÃO DO MODELO")
print("==============================")

print(f"Acurácia: {acuracia * 100:.2f}%")

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes_teste))


# ==========================================================
# 6. FUNÇÃO PARA MOSTRAR A ÁRVORE
# ==========================================================

def mostrar_arvore():

    plt.figure(figsize=(20, 10))

    plot_tree(
        modelo,
        feature_names=x.columns,
        class_names=modelo.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )

    plt.title("Árvore de Decisão - Desempenho dos Alunos")

    plt.show()


# ==========================================================
# 7. FUNÇÃO PARA MOSTRAR MATRIZ DE CONFUSÃO
# ==========================================================

def mostrar_matriz_confusao():

    matriz = confusion_matrix(
        y_teste,
        previsoes_teste,
        labels=modelo.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    imagem = ax.imshow(matriz, cmap="Blues")

    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Valor Real")

    ax.set_xticks(range(len(modelo.classes_)))
    ax.set_yticks(range(len(modelo.classes_)))

    ax.set_xticklabels(modelo.classes_)
    ax.set_yticklabels(modelo.classes_)

    # Mostra os números dentro da matriz
    for i in range(len(modelo.classes_)):
        for j in range(len(modelo.classes_)):
            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
                color="black"
            )

    plt.colorbar(imagem)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 8. FUNÇÃO DE PREVISÃO
# ==========================================================

def prever():

    try:

        horas = float(entry_horas.get())
        faltas_aluno = float(entry_faltas.get())
        nota_aluno = float(entry_nota.get())

        # Validação
        if horas < 0 or horas > 24:
            messagebox.showerror(
                "Erro",
                "As horas de estudo devem estar entre 0 e 24."
            )
            return

        if faltas_aluno < 0 or faltas_aluno > 100:
            messagebox.showerror(
                "Erro",
                "O número de faltas deve estar entre 0 e 100."
            )
            return

        if nota_aluno < 0 or nota_aluno > 10:
            messagebox.showerror(
                "Erro",
                "A nota deve estar entre 0 e 10."
            )
            return


        # DataFrame do novo aluno
        novo_aluno = pd.DataFrame(
            [[horas, faltas_aluno, nota_aluno]],
            columns=[
                "Horas_de_estudo",
                "Faltas",
                "Nota"
            ]
        )


        # Previsão
        previsao = modelo.predict(novo_aluno)[0]

        # Probabilidade
        probabilidades = modelo.predict_proba(novo_aluno)[0]

        indice = list(modelo.classes_).index(previsao)

        confianca = probabilidades[indice] * 100


        # Atualizando resultado
        label_resultado.config(
            text=f"Situação: {previsao}",
            foreground=cor_resultado(previsao)
        )

        label_confianca.config(
            text=f"Confiança aproximada: {confianca:.1f}%"
        )


        # Detalhes
        label_detalhes.config(
            text=(
                f"Horas de estudo: {horas:.1f}\n"
                f"Faltas: {faltas_aluno:.0f}\n"
                f"Nota: {nota_aluno:.1f}"
            )
        )


    except ValueError:

        messagebox.showerror(
            "Erro",
            "Digite apenas números válidos."
        )


# ==========================================================
# 9. COR DO RESULTADO
# ==========================================================

def cor_resultado(resultado):

    if resultado == "Aprovado":
        return "#16a34a"

    elif resultado == "Recuperação":
        return "#d97706"

    else:
        return "#dc2626"


# ==========================================================
# 10. LIMPAR CAMPOS
# ==========================================================

def limpar():

    entry_horas.delete(0, tk.END)
    entry_faltas.delete(0, tk.END)
    entry_nota.delete(0, tk.END)

    label_resultado.config(
        text="Situação: --",
        foreground="#1f2937"
    )

    label_confianca.config(
        text="Confiança aproximada: --"
    )

    label_detalhes.config(
        text="Informe os dados do aluno."
    )


# ==========================================================
# 11. INTERFACE
# ==========================================================

janela = tk.Tk()

janela.title("Sistema de Análise de Alunos")
janela.geometry("650x600")
janela.configure(bg="#f1f5f9")


# ------------------------------
# Estilo
# ------------------------------

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "TButton",
    font=("Arial", 11, "bold"),
    padding=10
)

style.configure(
    "TLabel",
    background="#f1f5f9",
    font=("Arial", 11)
)


# ==========================================================
# CABEÇALHO
# ==========================================================

frame_titulo = tk.Frame(
    janela,
    bg="#1e3a8a",
    height=100
)

frame_titulo.pack(
    fill="x"
)

label_titulo = tk.Label(
    frame_titulo,
    text="📚 Análise de Desempenho Escolar",
    bg="#1e3a8a",
    fg="white",
    font=("Arial", 22, "bold")
)

label_titulo.pack(
    pady=(20, 5)
)

label_subtitulo = tk.Label(
    frame_titulo,
    text="Preencha os dados do aluno para realizar a previsão",
    bg="#1e3a8a",
    fg="#dbeafe",
    font=("Arial", 10)
)

label_subtitulo.pack()


# ==========================================================
# FORMULÁRIO
# ==========================================================

frame_form = tk.Frame(
    janela,
    bg="white",
    padx=30,
    pady=25
)

frame_form.pack(
    padx=40,
    pady=25,
    fill="x"
)


# Horas
label_horas = tk.Label(
    frame_form,
    text="Horas de estudo:",
    bg="white",
    font=("Arial", 11, "bold")
)

label_horas.grid(
    row=0,
    column=0,
    sticky="w",
    pady=10
)

entry_horas = ttk.Entry(
    frame_form,
    width=30
)

entry_horas.grid(
    row=0,
    column=1,
    pady=10,
    padx=20
)


# Faltas
label_faltas = tk.Label(
    frame_form,
    text="Quantidade de faltas:",
    bg="white",
    font=("Arial", 11, "bold")
)

label_faltas.grid(
    row=1,
    column=0,
    sticky="w",
    pady=10
)

entry_faltas = ttk.Entry(
    frame_form,
    width=30
)

entry_faltas.grid(
    row=1,
    column=1,
    pady=10,
    padx=20
)


# Nota
label_nota = tk.Label(
    frame_form,
    text="Nota:",
    bg="white",
    font=("Arial", 11, "bold")
)

label_nota.grid(
    row=2,
    column=0,
    sticky="w",
    pady=10
)

entry_nota = ttk.Entry(
    frame_form,
    width=30
)

entry_nota.grid(
    row=2,
    column=1,
    pady=10,
    padx=20
)


# ==========================================================
# BOTÕES
# ==========================================================

frame_botoes = tk.Frame(
    janela,
    bg="#f1f5f9"
)

frame_botoes.pack(
    pady=5
)


botao_prever = ttk.Button(
    frame_botoes,
    text="🔍 Analisar Aluno",
    command=prever
)

botao_prever.grid(
    row=0,
    column=0,
    padx=5
)


botao_limpar = ttk.Button(
    frame_botoes,
    text="🗑 Limpar",
    command=limpar
)

botao_limpar.grid(
    row=0,
    column=1,
    padx=5
)


botao_arvore = ttk.Button(
    frame_botoes,
    text="🌳 Ver Árvore",
    command=mostrar_arvore
)

botao_arvore.grid(
    row=0,
    column=2,
    padx=5
)


botao_matriz = ttk.Button(
    frame_botoes,
    text="📊 Matriz",
    command=mostrar_matriz_confusao
)

botao_matriz.grid(
    row=0,
    column=3,
    padx=5
)


# ==========================================================
# RESULTADO
# ==========================================================

frame_resultado = tk.Frame(
    janela,
    bg="white",
    padx=20,
    pady=20
)

frame_resultado.pack(
    padx=40,
    pady=20,
    fill="x"
)


label_resultado = tk.Label(
    frame_resultado,
    text="Situação: --",
    bg="white",
    fg="#1f2937",
    font=("Arial", 20, "bold")
)

label_resultado.pack(
    pady=5
)


label_confianca = tk.Label(
    frame_resultado,
    text="Confiança aproximada: --",
    bg="white",
    fg="#475569",
    font=("Arial", 11)
)

label_confianca.pack(
    pady=5
)


label_detalhes = tk.Label(
    frame_resultado,
    text="Informe os dados do aluno.",
    bg="white",
    fg="#64748b",
    font=("Arial", 10)
)

label_detalhes.pack(
    pady=5
)


# ==========================================================
# INFORMAÇÕES DO MODELO
# ==========================================================

label_modelo = tk.Label(
    janela,
    text=f"Modelo treinado com {len(df)} alunos | "
         f"Acurácia no teste: {acuracia * 100:.2f}%",
    bg="#f1f5f9",
    fg="#475569",
    font=("Arial", 9)
)

label_modelo.pack(
    pady=5
)


# ==========================================================
# EXECUTAR
# ==========================================================

janela.mainloop()
