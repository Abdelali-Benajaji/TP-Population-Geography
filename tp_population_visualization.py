# ============================================================
# TP : Introduction to Artificial Intelligence
# Title : Interactive Geographic Analysis of World Population
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("world_population.csv")

# Clean column names (replace spaces with underscores)
df.columns = df.columns.str.replace(" ", "_")

# Fill missing density values
df["Density_(per_km²)"] = df["Density_(per_km²)"].fillna(
    df["2022_Population"] / df["Area_(km²)"]
)

# Add log scale for better visualization
df["Log_Density"] = np.log10(df["Density_(per_km²)"] + 1)

# Merge Western Sahara into Morocco
morocco_mask = (df["CCA3"] == "MAR") | (df["CCA3"] == "ESH")
morocco_combined = df[morocco_mask].copy()

# Sum population columns and area
pop_cols = ["2022_Population", "2020_Population", "2015_Population", 
            "2010_Population", "2000_Population", "1990_Population", 
            "1980_Population", "1970_Population", "Area_(km²)"]

morocco_totals = morocco_combined[pop_cols].sum()

# Update Morocco row with combined data
df.loc[df["CCA3"] == "MAR", pop_cols] = morocco_totals.values

# Recalculate density for Morocco
df.loc[df["CCA3"] == "MAR", "Density_(per_km²)"] = (
    morocco_totals["2022_Population"] / morocco_totals["Area_(km²)"]
)

# Remove Western Sahara row
df = df[df["CCA3"] != "ESH"].reset_index(drop=True)

# Fill missing density values
df["Density_(per_km²)"] = df["Density_(per_km²)"].fillna(
    df["2022_Population"] / df["Area_(km²)"]
)

# Display basic information
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# ------------------------------------------------------------
# 2. Data Analysis
# ------------------------------------------------------------

# Total world population in 2022
total_population_2022 = np.sum(df["2022_Population"])
print("\nWorld Population in 2022:", int(total_population_2022))

# Top 10 most populated countries (2022)
top10 = df.sort_values("2022_Population", ascending=False).head(10)
print("\nTop 10 Most Populated Countries (2022):")
print(top10[["Country", "2022_Population"]])

# Population by continent
population_by_continent = df.groupby("Continent")["2022_Population"].sum()

# ------------------------------------------------------------
# 3. Visualization – Interactive Maps
# ------------------------------------------------------------

# 3.1 World Population Distribution Map (2022)
fig_population = px.choropleth(
    df,
    locations="CCA3",
    color="2022_Population",
    hover_name="Country",
    color_continuous_scale="Plasma",
    title="World Population Distribution (2022)"
)
fig_population.show()

# 3.2 World Population Density Map
fig_density = px.choropleth(
    df,
    locations="CCA3",
    color="Log_Density",
    hover_name="Country",
    hover_data={"Density_(per_km²)": ":.2f", "Log_Density": False},
    color_continuous_scale="Viridis",
    title="World Population Density (Log Scale)"
)
fig_density.show()

# ------------------------------------------------------------
# 4. Visualization – Charts
# ------------------------------------------------------------

# Population by continent (bar chart)
fig_continent = px.bar(
    population_by_continent,
    title="Population by Continent (2022)",
    labels={"value": "Population", "Continent": "Continent"}
)
fig_continent.show()

# ------------------------------------------------------------
# 5. Population Evolution (Case Study: Morocco)
# ------------------------------------------------------------

country_name = "Morocco"
country_data = df[df["Country"] == country_name]

years = np.array([1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022])
population_values = country_data[
    [
        "1970_Population", "1980_Population", "1990_Population",
        "2000_Population", "2010_Population", "2015_Population",
        "2020_Population", "2022_Population"
    ]
].values[0]

# Line chart for population growth
fig_growth = px.line(
    x=years,
    y=population_values,
    title=f"Population Growth in {country_name}",
    labels={"x": "Year", "y": "Population"}
)
fig_growth.show()

# ------------------------------------------------------------
# 6. Machine Learning – Population Prediction (Bonus)
# ------------------------------------------------------------

# Linear Regression Model
X = years.reshape(-1, 1)
y = population_values

model = LinearRegression()
model.fit(X, y)

# ------------------------------------------------------------
# 7. Interactive Scatter Map - Major Cities
# ------------------------------------------------------------

# Create a dataset of major cities with their coordinates
cities_data = {
    'City': ['Tokyo', 'Delhi', 'Shanghai', 'São Paulo', 'Mumbai', 'Beijing', 
             'Cairo', 'Dhaka', 'Mexico City', 'Osaka', 'Karachi', 'Chongqing',
             'Istanbul', 'Buenos Aires', 'Kolkata', 'Lagos', 'Manila', 'Rio de Janeiro',
             'Guangzhou', 'Los Angeles', 'Moscow', 'Paris', 'Bangkok', 'Jakarta',
             'London', 'Lima', 'Seoul', 'Bogotá', 'Chennai', 'Bangalore'],
    'Latitude': [35.6762, 28.7041, 31.2304, -23.5505, 19.0760, 39.9042,
                 30.0444, 23.8103, 19.4326, 34.6937, 24.8607, 29.4316,
                 41.0082, -34.6037, 22.5726, 6.5244, 14.5995, -22.9068,
                 23.1291, 34.0522, 55.7558, 48.8566, 13.7563, -6.2088,
                 51.5074, -12.0464, 37.5665, 4.7110, 13.0827, 12.9716],
    'Longitude': [139.6503, 77.1025, 121.4737, -46.6333, 72.8777, 116.4074,
                  31.2357, 90.4125, -99.1332, 135.5023, 67.0011, 106.9123,
                  28.9784, -58.3816, 88.3639, 3.3792, 120.9842, -43.1729,
                  113.2644, -118.2437, 37.6173, 2.3522, 100.5018, 106.8456,
                  -0.1278, -77.0428, 126.9780, -74.0721, 80.2707, 77.5946],
    'Population_Millions': [37.4, 32.9, 28.5, 22.6, 20.7, 20.5,
                           21.3, 22.0, 21.9, 19.1, 16.8, 16.4,
                           15.6, 15.4, 15.1, 14.9, 14.4, 13.7,
                           13.6, 13.2, 12.6, 11.2, 10.9, 10.8,
                           9.5, 11.2, 9.9, 11.3, 11.5, 12.8],
    'Country': ['Japan', 'India', 'China', 'Brazil', 'India', 'China',
                'Egypt', 'Bangladesh', 'Mexico', 'Japan', 'Pakistan', 'China',
                'Turkey', 'Argentina', 'India', 'Nigeria', 'Philippines', 'Brazil',
                'China', 'USA', 'Russia', 'France', 'Thailand', 'Indonesia',
                'UK', 'Peru', 'South Korea', 'Colombia', 'India', 'India']
}

cities_df = pd.DataFrame(cities_data)

# Create interactive scatter map
fig_cities = px.scatter_map(
    cities_df,
    lat="Latitude",
    lon="Longitude",
    size="Population_Millions",
    color="Population_Millions",
    hover_name="City",
    hover_data={"Country": True, "Population_Millions": ":.1f", 
                "Latitude": False, "Longitude": False},
    color_continuous_scale="Turbo",
    size_max=30,
    zoom=1,
    title="World's Major Cities by Population",
    map_style="carto-positron"
)
fig_cities.show()

# Predict population in 2030
prediction_2030 = model.predict([[2030]])
print(f"\nPredicted population of {country_name} in 2030:",
      int(prediction_2030[0]))

# ------------------------------------------------------------
# End of TP
# ------------------------------------------------------------