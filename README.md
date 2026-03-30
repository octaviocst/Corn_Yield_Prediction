# 🌽 Predição de Produtividade de Milho com Machine Learning e Sensoriamento Remoto

[![Artigo Publicado](https://img.shields.io/badge/Publicado-Smart%20Agricultural%20Technology%202025-green)](https://doi.org/10.1016/j.atech.2025.101216)
[![Elsevier](https://img.shields.io/badge/Editora-Elsevier-orange)](https://doi.org/10.1016/j.atech.2025.101216)
[![R](https://img.shields.io/badge/Linguagem-R-276DC3?logo=r)](https://www.r-project.org/)

> Projeto de mestrado — UFLA (Universidade Federal de Lavras)

---

## 📌 Sobre o Projeto

Este repositório contém os dados e scripts de análise utilizados no artigo:

> **da Costa, O.P. et al. (2025)**
> *Impact of atmospheric corrections on satellite imagery for corn yield prediction using machine learning.*
> Smart Agricultural Technology, Elsevier. [https://doi.org/10.1016/j.atech.2025.101216](https://doi.org/10.1016/j.atech.2025.101216)

O estudo avalia o impacto de diferentes métodos de correção atmosférica em imagens de satélite para a predição de produtividade de milho com algoritmos de Machine Learning, com foco em encontrar o pipeline mais eficiente para aplicação em escala.

---

## 🎯 Objetivos

- Comparar métodos de correção atmosférica (DOS, iCOR e Sen2COR) em imagens de satélite
- Extrair índices de vegetação espectrais (NDVI, EVI, SAVI, etc.) como features
- Treinar e validar modelos de ML para predição de produtividade de milho
- Identificar o método de pré-processamento que maximiza a acurácia preditiva

---

## 🗂️ Estrutura do Repositório

```
corn-yield-prediction/
│
├── data/
│   ├── field_data/          # Dados de produtividade coletados em campo
│   ├── spectral_indices/    # Índices espectrais extraídos por correção atmosférica
│   └── README_data.md       # Descrição das variáveis e metodologia de coleta
│
├── code/
|   ├── R/          ← scripts em R
|   ├── python/     ← scripts em Python
│
├── results/
│   ├── figures/             # Gráficos e visualizações gerados
│   └── model_metrics/       # Métricas de desempenho dos modelos
│
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas

| Ferramenta | Uso |
|-----------|-----|
| **R e Python** | Linguagens de análise |
| **Pandas, Numpy e tidyverse** | Manipulação e visualização de dados |
| **Scikit-Learn e caret** | Treinamento e validação de modelos de ML |
| **randomForest / xgboost** | Algoritmos de predição |
| **Google Earth Engine** | Aquisição e pré-processamento de imagens de satélite |
| **SNAP ESA** | Correções Atmosféricas |
| **QGIS** | Processamento geoespacial |

---

## 🔬 Metodologia Resumida

1. **Coleta de dados de campo** — produtividade de milho (kg/ha) em parcelas experimentais
2. **Aquisição de imagens de satélite** — Sentinel-2 e Landsat nas fases fenológicas críticas
3. **Correções atmosféricas** — comparação de diferentes métodos de correção
4. **Extração de índices espectrais** — NDVI, EVI, SAVI, NDRE, entre outros
5. **Feature engineering** — seleção de variáveis e análise de correlação
6. **Modelagem** — Random Forest, SVM, XGBoost com validação cruzada k-fold
7. **Avaliação** — R², RMSE, MAE por algoritmo e por método de correção

---

## 📊 Principais Resultados

> Os resultados completos estão disponíveis no artigo publicado.
> Acesse em: [https://doi.org/10.1016/j.atech.2025.101216](https://doi.org/10.1016/j.atech.2025.101216)

---

## 📖 Como Citar

```bibtex
@article{da_Costa_2025,
  title   = {Impact of atmospheric corrections on satellite imagery for corn yield prediction using machine learning},
  volume  = {12},
  ISSN    = {2772-3755},
  DOI     = {10.1016/j.atech.2025.101216},
  journal = {Smart Agricultural Technology},
  publisher = {Elsevier BV},
  author  = {da Costa, Octávio Pereira and Inácio, Franklin Daniel and da Silva, Jéssica Elaine and Barboza, Thiago Orlando Costa and da Silva, Wender Henrique Batista and Lacerda, Lorena Nunes and dos Santos, Adão Felipe},
  year    = {2025},
  month   = dec,
  pages   = {101216}
}
```

---

## 📬 Contato

**Octávio Pereira da Costa**
📧 octavio.cst@gmail.com
🔗 [linkedin.com/in/octavio-costa-3b32a7b2](https://linkedin.com/in/octavio-costa-3b32a7b2)
🎓 UFLA — Universidade Federal de Lavras
