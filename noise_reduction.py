import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import export_svg

df = pd.read_csv('78F_forward_tilt.csv')

def identify_noise_points(df, col, noise_threshold):
    values = df[col]
    base_weight = values.mean()
    noise = np.abs(values - base_weight)
    noise_points = np.where(noise > noise_threshold * base_weight)[0]
    return noise_points

def plot_noisy_data(df, col, noise_indices, file_name):
    p = figure(x_axis_label='Index', y_axis_label='Weight (kg)')
    source = ColumnDataSource({
        'x': df.index,
        'y': df[col],
        'color': ['red' if i in noise_indices else 'blue' for i in df.index],
        'legend_label': ['Noise' if i in noise_indices else 'Data' for i in df.index]
    })
    p.scatter('x', 'y', color='color', source=source, fill_alpha=0.6, size=7, legend_field='legend_label')
    hover = HoverTool()
    hover.tooltips = [("Index", "@x"), ("Weight", "@y"), ("Type", "@legend_label")]
    p.add_tools(hover)
    p.legend.location = 'top_right'
    p.output_backend = "svg"
    export_svg(p, filename=f"{file_name}.svg")
    return p

def interpolate_and_adjust(df, noise_threshold):
    adjusted_df = df.copy()
    for col in adjusted_df.columns:
        noise_indices = identify_noise_points(adjusted_df, col, noise_threshold)
        data_copy = adjusted_df[col].copy()
        data_copy.iloc[noise_indices] = np.nan
        adjusted_df[col] = data_copy.interpolate(method='linear')
        plot_noisy_data(df, col, noise_indices, f'noisy_data_{col}')

    for index, row in adjusted_df.iterrows():
        row_total = row.sum()
        while row_total > 78 or row_total < 75:
            for i, value in enumerate(row):
                adjustment = (row_total - 78 if row_total > 78 else 75 - row_total) / len(row)
                row[i] += -adjustment if row_total > 78 else adjustment
            row_total = row.sum()
        adjusted_df.loc[index] = row

    return adjusted_df

cleaned_df = interpolate_and_adjust(df, 0.3)
cleaned_df['row_sum'] = cleaned_df.sum(axis=1)
assert cleaned_df['row_sum'].between(75, 78).all(), "Row totals not within the specified limits"

cleaned_df.to_csv('noise_cleaned_data_78.csv', index=True)

for col in cleaned_df.columns:
    plot_noisy_data(cleaned_df, col, [], f'filtered_data_{col}')