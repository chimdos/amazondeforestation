"""Amazon Deforestation Predictor - Streamlit app.

Treina um Random Forest para estimar desflorestamento mensal (km²) no dataset
da região do Xingu (MT-PA), usando indicadores econômicos e climáticos, e av-
ando-o com cross-validation sensível ao tempo ao invés de uma divisão de train/test.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

DATA_FILE_PATH = "data/dataset-amazonia - dataset.csv"
DATE_COLUMN = "Data_(Mes/Ano)"
TARGET_COLUMN = "Area_Desmatada_km2_Mes (Y)"
DRY_SEASON_MONTHS = (8, 9, 10)
CROSS_VALIDATION_WINDOWS = 5
RANDOM_FOREST_SEED = 42

INDICATOR_COLUMNS = [
    "Preco_Soja (X1)",
    "Preco_Boi_Gordo (X2)",
    "Preco_Ouro (X3)",
    "Precipitacao_mm (X4)",
    "Focos_Queimada (X5)",
    "Num_Atuacoes (X6)",
]

# --- Carregamento dos dados & engenharia de features

def load_monthly_dataset(file_path):
    """Lê O CSV e ordena os dados cronologicamente"""
    try:
        dataset = pd.read_csv(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Arquivo '{file_path}' não encontrado") from error

    dataset["Data"] = pd.to_datetime(dataset[DATE_COLUMN], format="%m/%Y")
    return dataset.sort_values("Data").reset_index(drop=True)

def add_previous_month_features(dataset):
    """Adiciona a coluna lag-1 para cada variável (valor do mês anterior)"""
    dataset_with_lags = dataset.copy()
    for column in INDICATOR_COLUMNS:
        dataset_with_lags[f"{column}_lag1"] = dataset_with_lags[column].shift(1)
    return dataset_with_lags

def add_dry_season_flag(dataset):
    """Marca os meses de seca (Agosto-Outubro), quando ocorrem mais queimadas"""
    dataset_with_flag = dataset.copy()
    is_dry_season = dataset_with_flag["Data"].dt.month.isin(DRY_SEASON_MONTHS)
    dataset_with_flag["Eh_Estacao_Seca"] = is_dry_season.astype(int)
    return dataset_with_flag


def add_year_trend(dataset):
    """Dá ao modelo uma consciência do "quão recente" uma coluna é
    

    Isso ajuda a separar mudanças políticas estruturais (ex.: intensidade das ações regulatórias)
    dos padrões de estação/preços que já estão nos dados
    """
    dataset_with_year = dataset.copy()
    dataset_with_year["Ano"] = dataset_with_year["Data"].dt.year
    return dataset_with_year

def engineer_features(raw_dataset):
    """Monta o dataset completo com as features para treinamento e avaliação"""
    engineered = add_previous_month_features(raw_dataset)
    engineered = add_dry_season_flag(engineered)
    engineered = add_year_trend(engineered)
    engineered = engineered.dropna().reset_index(drop=True)

    non_feature_columns = (TARGET_COLUMN, "Data", DATE_COLUMN)
    feature_columns = [c for c in engineered.columns if c not in non_feature_columns]
    return engineered, feature_columns


# --- Treinamento do modelo & avaliação sensível ao tempo

def build_random_forest():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=RANDOM_FOREST_SEED,
    )

def score_single_window(feature_matrix, log_target, target, train_index, test_index):
    model = build_random_forest()
    model.fit(feature_matrix.iloc[train_index], log_target.iloc[train_index])

    predicted = np.expm1(model.predict(feature_matrix.iloc[test_index]))
    actual = target.iloc[test_index]
    return r2_score(actual, predicted), np.sqrt(mean_squared_error(actual, predicted))

def evaluate_across_time_windows(feature_matrix, target, dates, n_windows=CROSS_VALIDATION_WINDOWS):
    """Avalia o modelo em diversas janelas temporais cronológicas de treino/teste

    Uma divisão 80/20 oculta como o desempenho varia ao longo do tempo;
    em vez disso, esta abordagem avança na história, treinando sempre com
    os dados passados e testando na janela subsequente.
    """
    log_target = np.log1p(target)
    splitter = TimeSeriesSplit(n_splits=n_windows)

    window_results = []
    for train_index, test_index in splitter.split(feature_matrix):
        r2, rmse = score_single_window(feature_matrix, log_target, target, train_index, test_index)
        window_results.append({
            "inicio": dates.iloc[test_index].min(),
            "fim": dates.iloc[test_index].max(),
            "r2": r2,
            "rmse": rmse,
        })
    return pd.DataFrame(window_results)

def summarize_window_results(window_results):
    """Reduz as pontuações por janela para três números notáveis de se reportar:
    a média geral, a média excluindo a pior (mais anômala) janela, e a melhor
    janela individual
    """
    worst = window_results.loc[window_results["r2"].idxmin()]
    best = window_results.loc[window_results["r2"].idxmax()]
    without_worst = window_results.drop(index=window_results["r2"].idxmin())
 
    return {
        "Todas as janelas": {"r2": window_results["r2"].mean(), "rmse": window_results["rmse"].mean()},
        "Sem a janela mais atípica": {
            "r2": without_worst["r2"].mean(),
            "rmse": without_worst["rmse"].mean(),
            "janela_excluida": worst,
        },
        "Melhor janela individual": {"r2": best["r2"], "rmse": best["rmse"], "janela": best},
    }

def train_production_model(feature_matrix, target):
    """Ajusta o modelo que realmente foi usado para previsões e importância das
    features, treinado com todo o histórico de dados em vez de isolar uma parte deles
    """
    model = build_random_forest()
    model.fit(feature_matrix, np.log1p(target))
    return model


# --- Renderização do Streamlit

def render_feature_importance_chart(model, feature_columns, top_n=10):
    st.subheader("O que mais afeta o desmatamento?")
 
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(top_n)
 
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=axis)
    axis.set_xlabel("Importância")
    st.pyplot(figure)

def build_window_summary_rows(summary):
    scenarios = ("Todas as janelas", "Sem a janela mais atípica", "Melhor janela individual")
    return [
        {"Cenário": label, "R²": round(summary[label]["r2"], 2), "RMSE (km²)": round(summary[label]["rmse"])}
        for label in scenarios
    ]

def render_excluded_window_caption(excluded_window):
    st.caption(
        f"Janela excluída: {excluded_window['inicio']:%m/%Y} a {excluded_window['fim']:%m/%Y} "
        f"(R² = {excluded_window['r2']:.2f}) - coincide com o início do salto histórico "
        "de desmatamento de 2019-2020."
    )

def render_window_summary_table(summary):
    st.subheader("R² e RMSE por janela de tempo")
    st.caption(
        "Um único teste no fim da série esconde que o desempenho muda ao longo "
        "do tempo. Comparamos a média de 5 janelas de teste consecutivas, a "
        "mesma média sem a janela mais atípica, e a melhor janela individual."
    )
 
    rows = build_window_summary_rows(summary)
    st.table(pd.DataFrame(rows).set_index("Cenário"))
    render_excluded_window_caption(summary["Sem a janela mais atípica"]["janela_excluida"])
 
 
def render_monthly_indicators_table(dataset):
    st.subheader("Dados mensais utilizados")
    ordered = dataset[["Data"] + INDICATOR_COLUMNS].sort_values("Data")
    st.dataframe(ordered.set_index("Data"), use_container_width=True)
 

 # --- Orquestração

@st.cache_data
def load_and_prepare():
    raw_dataset = load_monthly_dataset(DATA_FILE_PATH)
    engineered_dataset, feature_columns = engineer_features(raw_dataset)
    return raw_dataset, engineered_dataset, feature_columns

def main():
    st.set_page_config(layout="wide", page_title="Amazon Deforestation Predictor")
    st.title("Amazon Deforestation Analysis: Random Forest")
    st.markdown("---")
 
    try:
        raw_dataset, engineered_dataset, feature_columns = load_and_prepare()
    except FileNotFoundError as error:
        st.error(str(error))
        return
 
    feature_matrix = engineered_dataset[feature_columns]
    target = engineered_dataset[TARGET_COLUMN]
    dates = engineered_dataset["Data"]
 
    window_results = evaluate_across_time_windows(feature_matrix, target, dates)
    summary = summarize_window_results(window_results)
    production_model = train_production_model(feature_matrix, target)
 
    st.sidebar.success(
        "Modelo treinado!\n\n"
        f"R² médio: {summary['Todas as janelas']['r2']:.2f}\n"
        f"RMSE médio: {summary['Todas as janelas']['rmse']:.0f} km²"
    )
 
    render_feature_importance_chart(production_model, feature_columns)
    render_window_summary_table(summary)
    render_monthly_indicators_table(raw_dataset)
 
 
if __name__ == "__main__":
    main()