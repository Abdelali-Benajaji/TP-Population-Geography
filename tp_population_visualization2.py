# ============================================================
# TP : Introduction to Artificial Intelligence
# Title : Interactive Geographic Analysis of World Population
# Rebuilt with: Polars + JAX (instead of Pandas + NumPy)
# ============================================================

import polars as pl
import jax.numpy as jnp
from jax import device_put
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

df = pl.read_csv("world_population.csv")

# Clean column names (replace spaces with underscores)
df = df.rename({col: col.replace(" ", "_") for col in df.columns})

# Fill missing density values
df = df.with_columns([
    pl.when(pl.col("Density_(per_km²)").is_null())
    .then(pl.col("2022_Population") / pl.col("Area_(km²)"))
    .otherwise(pl.col("Density_(per_km²)"))
    .alias("Density_(per_km²)")
])

# Add log scale for better visualization
df = df.with_columns([
    (pl.col("Density_(per_km²)") + 1).log10().alias("Log_Density")
])

# Merge Western Sahara into Morocco
morocco_mask = (pl.col("CCA3") == "MAR") | (pl.col("CCA3") == "ESH")
morocco_combined = df.filter(morocco_mask)

# Sum population columns and area
pop_cols = ["2022_Population", "2020_Population", "2015_Population", 
            "2010_Population", "2000_Population", "1990_Population", 
            "1980_Population", "1970_Population", "Area_(km²)"]

morocco_totals = morocco_combined.select(pop_cols).sum()

# Update Morocco row with combined data
for col in pop_cols:
    df = df.with_columns([
        pl.when(pl.col("CCA3") == "MAR")
        .then(pl.lit(morocco_totals[col][0]))
        .otherwise(pl.col(col))
        .alias(col)
    ])

# Recalculate density for Morocco
morocco_pop = morocco_totals["2022_Population"][0]
morocco_area = morocco_totals["Area_(km²)"][0]

df = df.with_columns([
    pl.when(pl.col("CCA3") == "MAR")
    .then(pl.lit(morocco_pop / morocco_area))
    .otherwise(pl.col("Density_(per_km²)"))
    .alias("Density_(per_km²)")
])

# Remove Western Sahara row
df = df.filter(pl.col("CCA3") != "ESH")

# Recalculate Log_Density after Morocco merge
df = df.with_columns([
    (pl.col("Density_(per_km²)") + 1).log10().alias("Log_Density")
])

# Display basic information
print("Dataset Preview:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Types:")
print(df.schema)

# ------------------------------------------------------------
# 2. Data Analysis
# ------------------------------------------------------------

# Total world population in 2022 (using JAX)
# Filter out null values before converting to JAX array
population_series = df["2022_Population"].drop_nulls()
population_array = jnp.array(population_series.to_numpy())
total_population_2022 = jnp.sum(population_array)
print("\nWorld Population in 2022:", int(total_population_2022))

# Top 10 most populated countries (2022)
top10 = df.sort("2022_Population", descending=True).head(10)
print("\nTop 10 Most Populated Countries (2022):")
print(top10.select(["Country", "2022_Population"]))

# Population by continent
population_by_continent = df.group_by("Continent").agg([
    pl.col("2022_Population").sum()
]).sort("2022_Population", descending=True)

# ------------------------------------------------------------
# 3. Visualization – Interactive Maps
# ------------------------------------------------------------

# Convert to pandas for Plotly compatibility
df_pandas = df.to_pandas()

# 3.1 World Population Distribution Map (2022)
fig_population = px.choropleth(
    df_pandas,
    locations="CCA3",
    color="2022_Population",
    hover_name="Country",
    color_continuous_scale="Plasma",
    title="World Population Distribution (2022)"
)
fig_population.show()

# 3.2 World Population Density Map
fig_density = px.choropleth(
    df_pandas,
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
pop_by_continent_pandas = population_by_continent.to_pandas()
pop_by_continent_pandas = pop_by_continent_pandas.set_index("Continent")

fig_continent = px.bar(
    pop_by_continent_pandas,
    title="Population by Continent (2022)",
    labels={"value": "Population", "Continent": "Continent"}
)
fig_continent.show()

# ------------------------------------------------------------
# 5. Population Evolution (Case Study: Morocco)
# ------------------------------------------------------------

country_name = "Morocco"
country_data = df.filter(pl.col("Country") == country_name)

# Using JAX arrays for years and population
years = jnp.array([1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022])
population_values = jnp.array([
    country_data["1970_Population"][0],
    country_data["1980_Population"][0],
    country_data["1990_Population"][0],
    country_data["2000_Population"][0],
    country_data["2010_Population"][0],
    country_data["2015_Population"][0],
    country_data["2020_Population"][0],
    country_data["2022_Population"][0]
])

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

# Convert JAX arrays to NumPy for sklearn compatibility
X = jnp.array(years).reshape(-1, 1)
y = population_values

model = LinearRegression()
model.fit(X, y)

# ------------------------------------------------------------
# 7. Interactive Scatter Map - Major Cities
# ------------------------------------------------------------

# Create a dataset of major cities using Polars
cities_df = pl.DataFrame({
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
})

# Convert to pandas for Plotly
cities_pandas = cities_df.to_pandas()

# Create interactive scatter map
fig_cities = px.scatter_map(
    cities_pandas,
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

# Predict population in 2030 using JAX array
prediction_year = jnp.array([[2030]])
prediction_2030 = model.predict(prediction_year)
print(f"\nPredicted population of {country_name} in 2030:",
      int(prediction_2030[0]))

# ------------------------------------------------------------
# End of TP - Polars + JAX Version
# ------------------------------------------------------------