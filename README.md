# ML From Scratch

Implementações de algoritmos de Machine Learning usando apenas NumPy, com
foco em entender a matemática por trás de cada método. Cada algoritmo tem um
notebook com a derivação, experimentos e comparação com o scikit-learn.

## Projetos

| # | Algoritmo | O que explorei | Notebook |
|---|-----------|----------------|----------|
| 1 | Regressão Linear | Gradient descent, efeito da taxa de aprendizado e da padronização | [01_regressao_linear](notebooks/01_regressao_linear.ipynb) |
| 2 | Regressão Logística | Derivação da cross-entropy, fronteira de decisão, regularização | [02_regressao_logistica](notebooks/02_regressao_logistica.ipynb) |
| 3 | K-Means | Sensibilidade à inicialização, método do cotovelo, limitações | [03_kmeans](notebooks/03_kmeans.ipynb) |
| 4 | MLP | Backpropagation manual no MNIST | *em andamento* |

## Como rodar

```bash
git clone https://github.com/eduzx-afk/ml-from-scratch.git
cd ml-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Estrutura

- `src/`: implementações dos modelos
- `notebooks/`: derivações, experimentos e análises
