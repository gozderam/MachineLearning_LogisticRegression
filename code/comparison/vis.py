import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_comparison_results(df: pd.DataFrame):

    df["model"] = pd.Categorical(df['model'], ["log_reg_irls", "log_reg_gd", "log_reg_sgd", "log_reg_adam", "lda", "qda", "knn"])
    df = df.sort_values(by="model")

    fig = make_subplots(rows=2, cols=2, start_cell="top-left", subplot_titles=['Accuracy [%]', 'F1 Score', 'Precision', 'Recall'])

    fig.add_trace(go.Bar(x=df['model'], y=df['accuracy'], text = round(df['accuracy'], 2)),
        row=1, col=1)

    fig.add_trace(go.Bar(x=df['model'], y=df['f1_score'], text = round(df['f1_score'], 2)),
        row=1, col=2)

    fig.add_trace(go.Bar(x=df['model'], y=df['precision'], text = round(df['precision'], 2)),
        row=2, col=1)

    fig.add_trace(go.Bar(x=df['model'], y=df['recall'], text = round(df['recall'], 2)),
        row=2, col=2)

    fig.update_yaxes(title_text="accuracy [%]", row=1, col=1)
    fig.update_yaxes(title_text="F1 score", row=1, col=2)
    fig.update_yaxes(title_text="precision", row=2, col=1)
    fig.update_yaxes(title_text="recall", row=2, col=2)


    fig.update_layout(font_size=16,
        width=1000, height=800, title = { "text": "Comparison of models", "x" : 0.5}, showlegend=False)

    fig.show()
