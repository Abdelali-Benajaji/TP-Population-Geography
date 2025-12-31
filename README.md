# Interactive Geographic Analysis of World Population

This repository contains a small interactive visualization and analysis project for world population data, designed for an introductory AI / data-visualization assignment.

Files
- `tp_population_visualization.py` — Main script that loads `world_population.csv`, performs basic analysis, creates interactive Plotly visualizations (choropleth maps, bar and line charts), and fits a simple linear regression model to predict a country's population in 2030 (example uses Morocco).
- `world_population.csv` — Dataset containing population and density information for countries. The script expects columns such as country names, CCA3 codes, population at several years (e.g., `1970_Population`, `1980_Population`, ..., `2022_Population`), and `Density_(per_km²)`.
- `requirements.txt` — Python dependencies required to run the project.

Project overview
----------------
The script demonstrates:

- Loading and cleaning tabular data with pandas.
- Basic aggregation (world total population, population by continent, top-10 countries).
- Interactive geographic visualizations using Plotly Express (choropleth maps for population and density).
- Time series visualization for population growth (example: Morocco).
- A simple machine learning example using scikit-learn's `LinearRegression` to predict population for 2030 based on historical points.

Prerequisites
-------------
- Python 3.8+ (the script uses pandas, numpy, plotly, scikit-learn).
- A terminal (PowerShell recommended on Windows).

Install dependencies
--------------------
Install required packages from `requirements.txt`. From the repository root (PowerShell):

```powershell
python -m pip install --upgrade pip; 
pip install -r .\requirements.txt
```

Run the script
--------------
Run the main script from the repository root. The script will print dataset info and open interactive Plotly windows in your browser (or an inline window depending on environment).

```powershell
python .\tp_population_visualization.py
```

Notes about the run
- The script uses Plotly Express figures and calls `fig.show()` — on many systems this opens figures in your default web browser.
- Ensure `world_population.csv` is present in the same directory as the script.

Dataset / columns
-----------------
`world_population.csv` should contain at least the following columns (names used in the script):

- `Country` — country name.
- `CCA3` — 3-letter ISO country code used by Plotly for mapping.
- `Continent` — continent name for grouping.
- `1970_Population`, `1980_Population`, `1990_Population`, `2000_Population`, `2010_Population`, `2015_Population`, `2020_Population`, `2022_Population` — historical population columns used for the Morocco case study and regression.
- `2022_Population` — used for choropleth and top-10 ranking.
- `Density_(per_km²)` — population density used in the density choropleth.

What the script does (mapping to code sections)
---------------------------------------------
1. Load Dataset — reads `world_population.csv`, cleans column names by replacing spaces with underscores, and prints a preview and info.
2. Data Analysis — prints total world population (sum of `2022_Population`) and the top 10 most populated countries in 2022.
3. Visualization — creates two choropleth maps:
   - World population distribution (2022) using `CCA3` and `2022_Population`.
   - World population density using `Density_(per_km²)`.
4. Charts — a bar chart for population aggregated by `Continent`.
5. Population evolution (Case Study: Morocco) — plots a line chart of historical population values for Morocco.
6. Machine Learning — fits a simple linear regression on selected year/population points for Morocco and predicts population in 2030.

Interpreting the ML example
---------------------------
- The script fits a basic linear regression on the provided discrete historical points. This is a simple baseline and does not model complex demographic dynamics such as changing birth/death rates, migration, or policy changes.
- Treat the 2030 prediction as a rough linear projection. For better forecasts consider time-series models (ARIMA, Prophet), demographic cohort-component models, or more features (fertility rates, net migration, etc.).

Assumptions and known limitations
---------------------------------
- The linear regression uses only year as the predictor, so projections are simplistic.
- The script expects clean, complete columns for the listed years for the selected case country. Missing values or different column names will need adapting.
- Choropleth maps rely on correct `CCA3` (ISO3) country codes. Inconsistent or missing codes will result in missing countries on the map.

Troubleshooting
---------------
- If Plotly maps do not render: ensure you have a browser available. If running in a headless server, consider exporting static images or using Plotly's offline renderer with file output (e.g., `fig.write_html('map.html')`).
- If install fails: verify your Python version and that pip is targeting the same interpreter used to run the script.

Recommended next steps / improvements
-------------------------------------
1. Add a CLI entrypoint (argparse) to choose country, years, or whether to show plots vs save them to files.
2. Add error handling for missing/invalid CSV columns and for missing `CCA3` codes.
3. Package visualizations to HTML files (using `fig.write_html(...)`) so results can be shared without running the script.
4. Replace the linear regression example with a simple time-series model (Prophet or ARIMA) or add cross-validation and uncertainty estimates.
5. Add tests (small fixtures) verifying script functions (data loading, aggregation outputs) using pytest.

License
-------
This repository is provided for educational purposes. Modify the license as appropriate for your use case.

Contact
-------
If you want changes to the README or additional project wiring (CLI, tests, packaging), open an issue or update the files accordingly.

---
Generated: README for `tp_population_visualization.py` project
