import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Amazonia Predictor v2")

def prepare_data(df_raw):
    df_proc = df_raw.copy()
    target = 'Area_Desmatada_km2_Mes (Y)'

    cols_to_transform = ['Preco_Soja (X1)', 'Preco_Boi_Gordo (X2)', 'Preco_Ouro (X3)', 'Precipitacao_mm (X4)', 'Num_Atuacoes (X6)', 'Focos_Queimada (X5)']

    for col in cols_to_transform:
        df_proc[f'{col}_lag1'] = df_proc[col].shift(1)
        df_proc[f'{col}_lag2'] = df_proc[col].shift(2)
        df_proc[f'{col}_media3m'] = df_proc[col].rolling(window=3).mean()

    df_proc['Eh_Estacao_Seca'] = df_proc['Data'].dt.month.isin([8,9,10]).astype(int)
    
    df_proc = df_proc.dropna()

    features = [c for c in df_proc.columns if c not in [target, 'Data', 'Data_(Mes/Ano)']]
    X = df_proc[features]
    y = df_proc[target]

    y_log = np.log1p(y)
    
    split_point = int(len(df_proc) * 0.8)

    X_train = X.iloc[:split_point]
    y_train = y_log.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y_log.iloc[split_point:]

    return X_train, X_test, y_train, y_test, features

@st.cache_data
def load_and_train_model(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Data'] = pd.to_datetime(df['Data_(Mes/Ano)'], format='%m/%Y')
        df = df.sort_values('Data')
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{file_path}' não encontrado.")
        return None, None, None, None
    
    X_train, X_test, y_train, y_test, features = prepare_data(df)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    y_pred_log = model_rf.predict(X_test)

    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred_log))
    rmse = np.sqrt(mean_squared_error (np.expm1(y_test), np.expm1(y_pred_log)))

    return model_rf, df, features, (r2, rmse)

st.title("🌲 Amazonia Predictor v2: Random Forest")
st.markdown("---")

file_path = "data/dataset-amazonia - dataset.csv"
model, df, features, metrics = load_and_train_model(file_path)

if model is not None:
    r2, rmse = metrics
    st.sidebar.success(f"✅ Modelo Treinado!\n\nR²: {r2:.2f}\nRMSE: {rmse:.0f} km²")

    st.subheader("O que mais afeta o desmatamento?")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette="viridis", ax=ax)
    st.pyplot(fig)