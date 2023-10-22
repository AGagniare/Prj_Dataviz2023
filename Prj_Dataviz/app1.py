# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Load the dataset
def load_data():
    # Load the dataset
    file_path = 'liste-des-sites-patrimoniaux-remarquables-spr.csv'
    data = pd.read_csv(file_path, delimiter=';')
    return data

# Use the load_data function to load the data into the DataFrame
df = load_data()

# Title of the app
st.title("A Journey Through France’s Cultural Heritage")

#------------------------------

# Sidebar for adding student information
st.sidebar.header('Student Information')

# Collecting student information
student_name = st.sidebar.text_input('', 'Arthur Gagniare')
student_grade = st.sidebar.text_input('','Efrei')
student_prj = st.sidebar.text_input('', '#Dataviz 2023')



# Adding a LinkedIn link
linkedin_url = "https://www.linkedin.com/in/arthur-gagniare-6799761b6"
st.sidebar.markdown(f"[View My LinkedIn Profile]({linkedin_url})")
#------------------------------

# Line chart: Number of SPRs across years of creation
st.subheader("Number of SPRs Over Years")
spr_dates = df['spr_initial_date_de_creation'].dropna().astype('datetime64[ns]')
spr_counts_by_year = spr_dates.dt.year.value_counts().sort_index()
st.line_chart(spr_counts_by_year)

# Bar chart: Number of SPRs by region
st.subheader("Number of SPRs by Region")
spr_by_region = df['region'].value_counts()
st.bar_chart(spr_by_region)

# Scatter chart: Relationship between population and number of SPRs in a commune
st.subheader("Relationship between Population and Number of SPRs in a Commune")
communes_grouped = df.groupby('commune').agg({'population':'mean', 'nombre_de_spr':'sum'}).dropna()
st.scatter_chart(communes_grouped, x='population', y='nombre_de_spr')

#------------------------------


# TODO: Map visualization (This will be added)

# Creating a placeholder DataFrame with latitude and longitude
placeholder_data = pd.DataFrame({
    'latitude': [48.8566, 48.8511, 48.8609],  
    'longitude': [2.3522, 2.3582, 2.3387]     
})

# Map visualization using the placeholder data
if st.checkbox('Show Map Visualization'):
    st.map(placeholder_data)

#------------------------------


# Function to plot top N communes with the most SPRs
def plot_top_communes(n):
    top_communes = df.groupby('commune')['nombre_de_spr'].sum().sort_values(ascending=False).head(n)
    plt.figure(figsize=(10, 5))
    top_communes.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Top {n} Communes with the Most SPRs')
    plt.xlabel('Commune')
    plt.ylabel('Number of SPRs')
    st.pyplot()


# Interactive element to choose the number of top communes to display
n = st.slider('Select the Number of Top Communes to Display', min_value=5, max_value=50, value=10)
plot_top_communes(n)

# Bar plot using Seaborn
st.subheader("Average Population in Regions with SPRs")
if st.checkbox('Show Bar Plot'):
    fig, ax = plt.subplots(figsize=(10, 5))
    average_population_by_region = df.groupby('region')['population'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=average_population_by_region, y='region', x='population', palette='viridis', ax=ax)
    ax.set_xlabel('Average Population')
    ax.set_ylabel('Region')
    ax.set_title('Average Population in Regions with SPRs')
    st.pyplot(fig)

# Scatter plot using Plotly
st.subheader("Relationship Between Date of Creation and Number of Plans in Vigour")
df['spr_initial_date_de_creation'] = pd.to_datetime(df['spr_initial_date_de_creation'], errors='coerce')
...

# Bar Chart for Distribution of SPR Initial Regime of Creation using Altair
st.subheader("Distribution of SPR Initial Regime of Creation")
creation_regime_count = df['spr_initial_regime_de_creation'].value_counts().reset_index()
creation_regime_count.columns = ['Regime of Creation', 'Count']

# Creating the bar chart using Altair
chart = alt.Chart(creation_regime_count).mark_bar().encode(
    x='Regime of Creation',
    y='Count',
    color='Regime of Creation',
    tooltip=['Regime of Creation', 'Count']
).interactive()

st.altair_chart(chart, use_container_width=True)


# Interactive slider to filter data by year of SPR creation
year_to_filter = st.slider('Year to Filter SPRs by Creation Date',
                           min_value=int(spr_counts_by_year.index.min()),
                           max_value=int(spr_counts_by_year.index.max()),
                           value=int(spr_counts_by_year.index.max()))

filtered_data = df[df['spr_initial_date_de_creation'].dt.year <= year_to_filter]
st.write(f"Number of SPRs created up to {year_to_filter}: {len(filtered_data)}")

# Visualize the filtered data with a bar chart
if st.checkbox('Show Bar Chart of SPRs Created Each Year Up to Selected Year'):
    plt.figure(figsize=(12, 6))
    filtered_counts_by_year = df.loc[df['spr_initial_date_de_creation'].dt.year <= year_to_filter,
                                     'spr_initial_date_de_creation'].dt.year.value_counts().sort_index()
    plt.bar(filtered_counts_by_year.index, filtered_counts_by_year.values, color='skyblue', edgecolor='black')
    plt.xlabel('Year')
    plt.ylabel('Count of SPRs Created')
    plt.title(f'Count of SPRs Created Each Year Up to {year_to_filter}')
    st.pyplot()

# Select box to select and display data for a specific region
selected_region = st.selectbox('Select a Region to Explore SPRs', df['region'].dropna().unique())
region_data = df[df['region'] == selected_region]
st.write(region_data[['commune', 'nombre_de_spr', 'spr_initial_date_de_creation', 'population']])

#------------------------------

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6

# Load the dataset
df = pd.read_csv('liste-des-sites-patrimoniaux-remarquables-spr.csv', delimiter=';')

# Group the data by region and calculate the sum of plans in vigour
plans_in_vigour = df.groupby('region')['nombre_de_plans_en_vigueur'].sum().reset_index()

# Convert the data to ColumnDataSource
source = ColumnDataSource(data=plans_in_vigour)

# Create the Bokeh figure
p = figure(x_range=plans_in_vigour['region'].tolist(), height=350, 
           title="Number of Plans in Vigour by Region (using bokeh)",
           toolbar_location=None, tools="")

# Add a bar renderer
p.vbar(x='region', top='nombre_de_plans_en_vigueur', width=0.9, source=source, legend_field="region",
       line_color='white', fill_color=factor_cmap('region', palette=Spectral6, factors=plans_in_vigour['region'].tolist()))

# Customize the plot
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.axis_label = "Region"
p.yaxis.axis_label = "Total Number of Plans in Vigour"
p.xaxis.major_label_orientation = 1.2

# Show the Bokeh plot
st.bokeh_chart(p)


#------------------------------

# Calculate the average population in each region
avg_population = df.groupby('region')['population'].mean().reset_index()

# Create the Altair plot
alt_chart = alt.Chart(avg_population).mark_bar().encode(
    x='region',
    y='population',
    color=alt.Color('region', legend=None),
    tooltip=['region', 'population']
).properties(
    title='Average Population of Communes with SPRs by Region (using altair)'
).interactive()

st.altair_chart(alt_chart, use_container_width=True)

#------------------------------

df['year'] = pd.to_datetime(df['spr_initial_date_de_creation'], errors='coerce').dt.year  # Extract the year

# Filter the data to include only valid years
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# Get the counts of SPRs for each region and year
heatmap_data = df.groupby(['region', 'year']).size().reset_index(name='count')

# Corrected pivot function
heatmap_data = heatmap_data.pivot(index='region', columns='year', values='count')

# Create a heatmap visualization
if st.checkbox('Show Heatmap of SPR Distribution Across Regions and Years'):
    st.write("Heatmap showing the count of SPRs across different regions and years.")
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap='viridis', linewidths=.5)
    st.pyplot()

#------------------------------

# TODO: France Map heatmap (This will be added)


import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Load the GeoJSON data of French departments from a URL
url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
gdf = gpd.read_file(url)

# Count the SPR per department
spr_per_department = df['departement'].value_counts().reset_index()
spr_per_department.columns = ['departement', 'spr_count']

# Merge the GeoDataFrame with the SPR count data on the department name
merged = gdf.merge(spr_per_department, left_on='nom', right_on='departement', how='left')

# Handle NaN values after the merge
merged['spr_count'].fillna(0, inplace=True)

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot the choropleth map with a specific color scale
merged.plot(column='spr_count', ax=ax, legend=True, cmap='OrRd', edgecolor='k', linewidth=0.5,
            legend_kwds={'label': "Number of SPRs by Department"},
            vmin=1, vmax=10000)  # Set the scale of the heatmap here

ax.set_axis_off()
plt.title('Number of Sites Patrimoniaux Remarquables (SPR) by French Department')

# Display the map in the Streamlit app
st.pyplot(fig)