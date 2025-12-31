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
    color="Density_(per_km²)",
    hover_name="Country",
    color_continuous_scale="Viridis",
    title="World Population Density"
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
].values.flatten()

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

# Predict population in 2030
prediction_2030 = model.predict([[2030]])
print(f"\nPredicted population of {country_name} in 2030:",
      int(prediction_2030[0]))

# ------------------------------------------------------------
# End of TP
# ------------------------------------------------------------
