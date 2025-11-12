# Projeto de Machine Learning - Classificação: Breast Cancer

**Objetivo:** Desenvolver uma aplicação em Python utilizando scikit-learn que resolve um problema de classificação (detecção de câncer de mama) aplicando validação cruzada e regularização.

## Integrantes
- Nome: ______________________ RM: ____________
- Nome: ______________________ RM: ____________

## Descrição do problema
Classificação binária utilizando o dataset "Breast Cancer Wisconsin" (sklearn.datasets.load_breast_cancer). O objetivo é predizer se um tumor é maligno ou benigno a partir de características extraídas de imagens.

## Estratégia proposta
1. Carregar o dataset do sklearn.
2. Pré-processamento: StandardScaler.
3. Modelos testados:
   - Regressão Logística com regularização L2 (GridSearchCV para buscar melhor C).
   - Random Forest como baseline (ajuste de n_estimators e max_depth via GridSearchCV).
4. Avaliação:
   - Validação cruzada estratificada (StratifiedKFold, 5 folds).
   - Métricas: acurácia, precisão, recall, f1-score, ROC AUC.
5. Persistir o melhor modelo e gerar relatório (PDF) com resultados e gráficos.

## Arquivos entregues
- `README.md` (este arquivo)
- `requirements.txt` (dependências)
- `train.py` (código fonte para treinamento, avaliação, geração de relatório)
- `report.pdf` (modelo de relatório - gerado automaticamente quando executar train.py)

## Como executar
1. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script de treinamento:
   ```bash
   python train.py
   ```
4. O script produzirá:
   - `outputs/best_model.pkl` (modelo treinado)
   - `outputs/report.pdf` (relatório com métricas e gráficos)
   - arquivos de log na pasta `outputs/`

## Observações
- O código utiliza apenas bibliotecas amplamente disponíveis (scikit-learn, pandas, matplotlib, fpdf, joblib).
- Ajustes e recursos extras (ex.: hyperparameter tuning mais intenso, pipeline com feature selection) podem ser adicionados conforme demanda.
