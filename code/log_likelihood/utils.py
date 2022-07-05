from plotly.subplots import make_subplots
import plotly.graph_objects as go

def visualize_log_likelihood(irls_estimator, gd_estimator, sgd_estimator, adam_estimator, dataset_name):
    irls_ll = irls_estimator["logistic_regression"].log_likelihood_
    gd_ll = gd_estimator["logistic_regression"].log_likelihood_
    sgd_ll = sgd_estimator["logistic_regression"].log_likelihood_
    adam_ll = adam_estimator["logistic_regression"].log_likelihood_

    fig = make_subplots(rows=2, cols = 2, subplot_titles=("IRLS", "GD", "SGD", "ADAM"))

    fig.add_trace(
        go.Scatter(x = list(range(irls_ll.shape[0])), y = irls_ll),
        row = 1, col = 1
    )

    fig.add_trace(
        go.Scatter(x = list(range(gd_ll.shape[0])), y = gd_ll),
        row = 1, col = 2
    )

    fig.add_trace(
        go.Scatter(x = list(range(sgd_ll.shape[0])), y = sgd_ll),
        row = 2, col = 1
    )
    #to change
    fig.add_trace(
        go.Scatter(x = list(range(adam_ll.shape[0])), y = adam_ll),
        row = 2, col = 2
    )

    fig.update_layout(
        font_size=16,
        width=1100, height=800,
        title={
            'text': f"Log-likelihood for {dataset_name} dataset",
            'x' : 0.5
        },
        showlegend = False,
        xaxis= { "title": "iteration" },
        xaxis1={ "title": "iteration" },
        xaxis2={ "title": "iteration" },
        xaxis3={ "title": "iteration" },
        xaxis4={ "title": "iteration" },
        yaxis1 ={ "title": "log-likelihood" },
        yaxis2 ={ "title": "log-likelihood" },
        yaxis3 ={ "title": "log-likelihood" },
        yaxis4 ={ "title": "log-likelihood" }
    )

    return fig

