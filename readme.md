# Central Bank Digital Currency (CBDC) Impact Simulation

This project simulates the impact of Central Bank Digital Currency (CBDC) on a financial network consisting of a central bank, commercial banks, and individuals (consumers and merchants). The simulation analyzes how introducing a CBDC affects network structure, bank centrality, transaction volumes, and banking system metrics.

## Features

- **Agent-Based Modeling**: Implements CentralBank, CommercialBank, and Individual agents using the Mesa framework.
- **Network Analysis**: Calculates centrality metrics (degree, betweenness, eigenvector) for banks and the overall network.
- **CBDC Simulation**: Compares scenarios with and without CBDC, tracking changes in deposits, loans, and transaction patterns.
- **Visualization**: Generates network graphs, metric histograms, time series plots, and heatmaps to visualize simulation results.
- **Data Export**: Saves simulation data and analysis results as CSV files for further study.

## Project Structure

- [`main.py`](main.py): Main script to run simulations, generate visualizations, and analyze results.
- [`model.py`](model.py): Defines the `FinancialNetworkModel` and its setup, agent creation, and step logic.
- [`agents.py`](agents.py): Implements agent classes: `CentralBank`, `CommercialBank`, and `Individual`.
- [`network_analysis.py`](network_analysis.py): Functions for calculating and comparing network metrics.
- [`visualization.py`](visualization.py): Functions for plotting network structure, metrics, and comparisons.
- [`utils.py`](utils.py): Utility functions for saving data, generating visualizations, and summarizing analysis.
- [`pyproject.toml`](pyproject.toml): Python dependencies.
- [`output/`](output/): Directory where simulation results and plots are saved.

## Requirements

- Python 3.11+
- mesa
- networkx
- numpy
- pandas
- matplotlib
- seaborn

Install dependencies with:

```sh
pip install mesa matplotlib networkx pandas numpy seaborn
```

## Running the Simulation

To run the simulation and generate results:

```sh
python main.py
```

This will:
- Run two simulations (with and without CBDC)
- Save data and visualizations to the `output/` directory
- Print a summary of the CBDC impact analysis

## Output

- CSV files with model and agent metrics, transaction history, and comparison summaries
- PNG images of network visualizations and metric plots
- Analysis summary printed to the console

## Customization

You can adjust simulation parameters (number of banks, individuals, steps, etc.) in [`main.py`](main.py) and [`model.py`](model.py).

## License

This project is for educational and research purposes.

---