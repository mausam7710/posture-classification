import pandas as pd
from sklearn.cluster import DBSCAN
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import row

df = pd.read_csv("noise_cleaned_data_78.csv")

model = DBSCAN(eps=0.4, min_samples=50)
model.fit(df[['row_sum']])
df['is_outlier'] = model.labels_ == -1
df['is_outlier'] = df['is_outlier'].astype(str)

df['color'] = ['blue' if outlier == 'False' else 'red' for outlier in df['is_outlier']]

df.to_csv("78F.csv", index=False)

source_with_outliers = ColumnDataSource(df)

outlier_tooltips = [("index", "$index"), ("Total Weight", "@row_sum{0.2f} kg"), ("Outlier", "@is_outlier")]

p1 = figure(title="With Outliers", x_axis_label='Row Index', y_axis_label='Total Weight (kg)')
p1.scatter('index', 'row_sum', size=8, source=source_with_outliers, color='color', legend_label="Data")

p1.scatter('index', 'row_sum', size=8, color="red", legend_label="Outliers")
p1.legend.location = "top_left"

tooltips = [("index", "$index"), ("Total Weight", "@row_sum{0.2f} kg"), ("Outlier", "@is_outlier")]
p1.add_tools(HoverTool(tooltips=tooltips))

df_filtered = df[df['is_outlier'] == 'False']
source_filtered = ColumnDataSource(df_filtered)

p2 = figure(title="Outliers Removed", x_axis_label='Row Index', y_axis_label='Total Weight (kg)')
filtered_scatter = p2.scatter('index', 'row_sum', size=8, source=source_filtered, color='blue', legend_label="Data")
p2.add_tools(HoverTool(tooltips=tooltips))

layout = row(p1, p2)
show(layout)
