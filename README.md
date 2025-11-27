# Amazônia Predictor

Este projeto é uma aplicação de Data Science e Machine Learning desenvolvida para analisar e prever o desmatamento na Amazônia Legal. Utiliza dados históricos socioeconômicos e ambientais para identificar tendências e os fatores de maior impacto na devastação da floresta.

O projeto culmina em um dashboard interativo construído com **Streamlit**, alimentado por um modelo **Random Forest**.

## Objetivo

O principal objetivo é fornecer uma ferramenta analítica que permita:
1.  **Prever** a área desmatada ($km^2$) para meses futuros com base em dados históricos.
2.  **Identificar** quais variáveis (ex: preço do ouro, chuvas, fiscalização) têm maior correlação com o aumento ou redução do desmatamento.
3.  **Visualizar** a performance do modelo e a importância das *features* de forma clara.

## Estrutura do Projeto

```text
amazondeforestation/
├── app/
│   └── main.py              # Aplicação Streamlit (Dashboard e Pipeline de Produção)
├── data/
│   └── dataset-amazonia - dataset.csv  # Dados históricos (2015-2025)
├── notebooks/
│   └── analise_exploratoria.ipynb      # Estudo, comparação de modelos e validação
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação