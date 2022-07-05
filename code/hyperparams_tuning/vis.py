import plotly.express as px
import pandas as pd
import numpy as np


def plot_acc_vs_hyperparameter(clfs, hyperparam_name, dataset_names, model_names):
    h_name = hyperparam_name.split('__')[1]
    hyperparam_name = f'param_{hyperparam_name}'
    df = __get_data(clfs, hyperparam_name, dataset_names, model_names)
    df = df.sort_values(by='x', ascending=True)
    color = 'dataset' if type(dataset_names) == list else 'model'
    title_begining = f"{', '.join(dataset_names)} datasets, {model_names} model" if type(dataset_names) == list else f'{dataset_names} dataset'
    fig = px.line(df, x='x', y='y', color=color, labels={
                        "x": f"{h_name} hyperparameter value",
                        "y": "avg. validation accuracy"
                    },
                    title=f'{title_begining}: {h_name} hyperparameter vs. avg. validation accuracy',
                    width=1000, height=500)
    fig.update_traces(marker_size=10)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        font_size=16
    )

    fig.show()


def __get_data(clfs, hyperparam_name, dataset_names, model_names):
    if type(clfs) != list:
        x = np.array(clfs.cv_results_[hyperparam_name], dtype=float)
        y = clfs.cv_results_['mean_test_score']
        data = {'x': x, 'y': y, 'model': model_names}
        return pd.DataFrame(data)
    else:
        df = pd.DataFrame()
        for i in range(0, len(clfs)):
            x = np.array(clfs[i].cv_results_[hyperparam_name], dtype=float)
            y = clfs[i].cv_results_['mean_test_score']
            if type(model_names) == list:
                model = model_names[i]
                data = {'x': x, 'y': y, 'model': model}
            else:
                dataset = dataset_names[i] 
                data = {'x': x, 'y': y, 'dataset': dataset}
            df_temp = pd.DataFrame(data)
            df = pd.concat([df, df_temp], ignore_index=True)
        return df

