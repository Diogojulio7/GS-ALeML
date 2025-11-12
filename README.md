# Projeto - Ferramentas de Monitoramento de Bem-Estar e Saúde Mental no Trabalho

**Curso / Instituição:** FIAP – Engenharia de Software 3ESPR
**Integrantes:**
- Diogo Julio - RM553837
- Jonata Rafael - RM552939

## Objetivo
Desenvolver um projeto de ciência de dados que demonstra como ferramentas e modelos podem apoiar o monitoramento de bem-estar e saúde mental no ambiente de trabalho. O projeto inclui geração de dados sintéticos representativos e três análises principais: classificação de risco de burnout, predição de humor diário (regressão) e clusterização de perfis de bem-estar.

## Estrutura do projeto
- `README.md` - este arquivo
- `requirements.txt` - dependências necessárias
- `train.py` - script principal que gera o dataset sintético e executa as 3 análises (classificação, regressão e clusterização)
- `outputs/` - pasta onde os resultados (modelos, gráficos e relatório) serão gerados após executar `train.py`
- `ML_Project_Wellbeing.zip` - pacote pronto para entrega (gerado já nesta pasta)

## Como usar
1. Criar e ativar um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate    # Windows
```
2. Instalar dependências:
```bash
pip install -r requirements.txt
```
3. Executar o script (gera outputs em `outputs/`):
```bash
python train.py
```
4. Arquivos gerados:
- `outputs/report.pdf` - relatório consolidado (gerado automaticamente pelo script)
- `outputs/best_models/` - modelos salvos para as tarefas (se aplicável)
- `outputs/*.png` - gráficos gerados (ROC, scatter, importância de features, etc.)

## Observações
- O dataset é sintético e foi modelado para ilustrar conceitos e técnicas; em um cenário real substitua pelo dataset real da empresa (com anonimização e conformidade com LGPD).
- O script contém comentários explicativos e seções separadas para cada análise para facilitar o entendimento.
