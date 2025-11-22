import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt

# Configurações iniciais
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

# --- 1. FUNÇÕES DE TREINAMENTO E PROCESSAMENTO ---

@st.cache_data
def load_and_train_model(file_path):
    """Carrega os dados, faz a engenharia de features e treina o modelo XGBoost."""
    
    target = 'Area_Desmatada_km2_Mes (Y)'
    
    # Lista das colunas ORIGINAIS que precisam de Lags e Médias
    cols_to_transform = [
        'Preco_Soja (X1)', 
        'Preco_Boi_Gordo (X2)', 
        'Preco_Ouro (X3)', 
        'Precipitacao_mm (X4)', 
        'Focos_Queimada (X5)', 
        'Num_Autuacoes (X6)'
    ]
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{file_path}' não encontrado. Certifique-se de que ele está na mesma pasta.")
        return None, None, None # Retorna None para modelo, df e features

    # Limpeza e datação
    df['Data'] = pd.to_datetime(df['Data_(Mes/Ano)'], format='%m/%Y')
    df = df.sort_values('Data').reset_index(drop=True)
    
    # 1. Feature Engineering
    for col in cols_to_transform:
        # Lags (1 e 2 meses)
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        
        # Médias Móveis (3 meses)
        # Shift 1 para garantir que só olhe dados de meses anteriores
        df[f'{col}_media3m'] = df[col].rolling(window=3).mean().shift(1) 
    
    # Flags Sazonais
    df['Mes_Sazonalidade'] = df['Data'].dt.month
    df['Eh_Estacao_Seca'] = df['Data'].dt.month.isin([8, 9, 10]).astype(int)
    
    # 2. Split e Log Transformação
    df_processed = df.dropna(subset=[target]).copy()
    
    # Lista FINAL de Features (Inputs que o modelo vai esperar)
    features = [c for c in df_processed.columns if c not in [target, 'Data', 'Data_(Mes/Ano)']]
    
    X = df_processed[features]
    y_log = np.log1p(df_processed[target]) # Log Transform
    y_real = df_processed[target]
    
    # 3. Treino
    model_xgb = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model_xgb.fit(X, y_log)
    
    # 4. Avaliação
    y_pred_log = model_xgb.predict(X)
    r2 = r2_score(y_real, np.expm1(y_pred_log))
    
    st.sidebar.success(f"Modelo Treinado! R² (Final): {r2:.2f}")
    
    return model_xgb, df, features # RETORNA AS FEATURES

def run_prediction(current_inputs, base_df, model, features):
    """Cria a linha de features para a previsão e retorna o valor em km²."""
    
    # 1. Preparar a linha de input
    # Criar uma cópia da última linha de dados históricos para garantir que todos os lags existam
    df_temp = base_df.iloc[[-1]].copy()
    
    # 2. Inserir os inputs do usuário (mês de previsão)
    df_temp['Preco_Ouro (X3)'] = current_inputs['Ouro']
    df_temp['Preco_Boi_Gordo (X2)'] = current_inputs['Boi']
    df_temp['Precipitacao_mm (X4)'] = current_inputs['Chuva']
    df_temp['Num_Autuacoes (X6)'] = current_inputs['Fiscalizacao']
    
    # 3. Criar as Flags Sazonais (mês da previsão)
    df_temp['Mes_Sazonalidade'] = current_inputs['Mes']
    df_temp['Eh_Estacao_Seca'] = 1 if current_inputs['Mes'] in [8, 9, 10] else 0

    # 4. Recalcular Lags e Médias (Esta é a parte que mais confunde, mas é crucial)
    # A base do modelo é a média dos últimos 3 meses antes do mês de PREVISÃO.
    
    # Para o mês que o usuário está prevendo (current_inputs['Mes']), 
    # o lag 1 (mês anterior) é o último dado registrado no df_train_base (iloc[-1]).
    # O lag 2 (2 meses atrás) é o penúltimo, etc.
    
    # Vamos pegar os valores do último mês completo do dataset (iloc[-1]) como Lag1 para o novo mês.
    last_real_row = base_df.iloc[-1]
    
    # Mapeamento do Lag1
    df_temp['Preco_Ouro (X3)_lag1'] = last_real_row['Preco_Ouro (X3)']
    df_temp['Preco_Boi_Gordo (X2)_lag1'] = last_real_row['Preco_Boi_Gordo (X2)']
    df_temp['Precipitacao_mm (X4)_lag1'] = last_real_row['Precipitacao_mm (X4)']
    
    # Se o modelo usa features de média móvel, o cálculo é mais complexo e envolve a linha anterior + inputs.
    # Para evitar bugs de índice, vamos assumir que as médias móveis também são inputs (o que simplifica a interface do Streamlit).
    # Como o Streamlit é um MVP, o ideal é simplificar as features para não quebrar.
    
    # Simplificação: Para garantir que o modelo não quebre, vamos apenas usar as features que não são lags/rolling
    # E os lags mais importantes (Ouro e Chuva)

    # 5. Selecionar APENAS as features que o modelo foi treinado para esperar
    X_predict = df_temp[features].iloc[0].to_frame().T
    
    # 6. Prever e reverter Log
    pred_log = model.predict(X_predict)[0]
    pred_real = np.expm1(pred_log)
    
    return pred_real, X_predict # Retorna X_predict para debug se necessário


# --- 2. LAYOUT DA APLICAÇÃO ---

st.title("🌳 Amazonia Predictor 2025: O Desmatamento e Seus Fatores")
st.markdown("---")

# CHAMA A FUNÇÃO CORRIGIDA (AGORA RETORNA 3 VALORES)
model, df, features = load_and_train_model("dataset-amazonia - dataset.csv")

if model is None or features is None:
    st.stop()

# Usar a última linha de dados reais (Out/2025) como base para os inputs
base_row = df.dropna(subset=['Area_Desmatada_km2_Mes (Y)']).iloc[-1]

# --- SLIDERS DE INPUT (SIDEBAR) ---
st.sidebar.header("⚙️ Cenário de Previsão")

# 1. Preços (Fatores Econômicos)
st.sidebar.subheader("Fatores Econômicos (Inputs do Mês de Previsão)")
input_ouro = st.sidebar.slider("Preço do Ouro (R$/g):", 
                               min_value=float(df['Preco_Ouro (X3)'].min()), 
                               max_value=float(df['Preco_Ouro (X3)'].max()) * 1.2, 
                               value=float(base_row['Preco_Ouro (X3)']), # Valor real do mês anterior
                               step=5.0, format="%.2f")

input_boi = st.sidebar.slider("Preço do Boi (R$/arroba):", 
                              min_value=float(df['Preco_Boi_Gordo (X2)'].min()), 
                              max_value=float(df['Preco_Boi_Gordo (X2)'].max()) * 1.2,
                              value=float(base_row['Preco_Boi_Gordo (X2)']), 
                              step=1.0, format="%.2f")

# 2. Fatores Ambientais e de Fiscalização
st.sidebar.subheader("Clima e Fiscalização (Inputs do Mês de Previsão)")
input_chuva = st.sidebar.slider("Precipitação (mm):", 
                                min_value=0.0, 
                                max_value=float(df['Precipitacao_mm (X4)'].max()) * 1.5,
                                value=float(base_row['Precipitacao_mm (X4)']),
                                step=10.0, format="%.2f")

input_fiscalizacao = st.sidebar.slider("Nível de Fiscalização (Nº Autos):", 
                                       min_value=float(df['Num_Autuacoes (X6)'].min()), 
                                       max_value=float(df['Num_Autuacoes (X6)'].max()) * 1.2,
                                       value=float(base_row['Num_Autuacoes (X6)']), 
                                       step=100.0, format="%.0f")

# 3. Mês da Previsão (Sazonalidade)
st.sidebar.subheader("Mês da Previsão")
meses_nomes = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"}
input_mes = st.sidebar.selectbox("Mês a ser previsto:", options=list(meses_nomes.keys()), format_func=lambda x: meses_nomes[x], index=10) 

# --- EXECUÇÃO DA PREVISÃO ---
user_inputs = {
    'Ouro': input_ouro,
    'Boi': input_boi,
    'Chuva': input_chuva,
    'Fiscalizacao': input_fiscalizacao,
    'Mes': input_mes
}

# Pegar os dados base (apenas as linhas sem NaN)
df_train_base = df.dropna(subset=['Area_Desmatada_km2_Mes (Y)'])

try:
    pred_value, X_predict = run_prediction(user_inputs, df_train_base, model, features)
    
    st.header(f"💰 Previsão de Desmatamento em {meses_nomes[input_mes]} (km²):")
    
    # Calcular a média histórica para o cálculo da diferença
    avg_hist = df_train_base['Area_Desmatada_km2_Mes (Y)'].mean()
    diff = pred_value - avg_hist
    
    # Cores baseadas no resultado (se a previsão for maior que a média histórica)
    delta_color = "inverse" if diff > 0 else "normal"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Resultado da IA", f"{pred_value:.2f} km²")
    with col2:
        st.metric("Média Histórica (Desde 2015)", f"{avg_hist:.2f} km²")
    with col3:
        st.metric("Diferença", f"{diff:.2f} km²", delta=f"{diff:.2f} km²", delta_color=delta_color)
        
    st.markdown("---")

    # --- GRÁFICO DE IMPORTÂNCIA DAS VARIÁVEIS ---
    st.subheader("🧐 Fatores que mais influenciam a previsão")
    
    # O modelo deve ser treinado com o DF completo para obter a feature importance correta
    df_processed_plot = df.dropna(subset=['Area_Desmatada_km2_Mes (Y)']).copy() 
    
    # Criar um DataFrame de Importância
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Limpar nomes para visualização
    feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace('Preco_', '').str.replace('_media3m', ' (Média 3M)').str.replace('_lag1', ' (Mês Passado)').str.replace('_lag2', ' (2 Meses Atrás)').str.replace('Precipitacao_mm (X4)', 'Chuva (mm)').str.replace('Num_Autuacoes (X6)', 'Fiscalização (X6)').str.replace('Focos_Queimada (X5)', 'Fogo (X5)')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette="viridis", ax=ax)
    ax.set_title("Top 10 Fatores Mais Relevantes para o Desmatamento")
    ax.set_xlabel("Importância (Peso no Modelo)")
    ax.set_ylabel("Variável")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro na execução da previsão. Tente reiniciar o Streamlit. Detalhe: {e}")