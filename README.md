# TailCoR Analysis

This project implements the **TailCoR** dependence measure for financial time series, based on the paper:

> Chiapino, F., Ortobelli Lozza, S., Rachev, S. T., & Sframe, S. (2022).  
> *TailCoR: A dependence measure for financial contagion*. PLOS ONE, 17(12), e0278599.  
> [https://doi.org/10.1371/journal.pone.0278599](https://doi.org/10.1371/journal.pone.0278599)

The program loads financial instruments data (CSV inside ZIP files), computes log returns, and calculates **TailCoR, linear, and nonlinear dependence matrices**.  
It also produces cluster heatmaps and plots the evolution of average TailCoR values over time.

---

## Features
- Load and preprocess stock price data from ZIP archives.
- Compute log returns for multiple instruments.
- Calculate:
  - TailCoR correlation matrix
  - Linear component
  - Nonlinear component
- Build combined dependence matrices across instruments.
- Rolling-window average TailCoR analysis.

---

## Requirements
- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `seaborn`

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

---

## Usage
1. Place your ZIP files containing CSV market data in a folder.  
   - Each CSV must include columns: `Name`, `Date`, `Close`.  
   - Dates should be in `YYYYMMDD` format.
2. Set the `folder_path` variable in the script to point to this folder.
3. Run the script:
   ```bash
   python tailcor_analysis.py
   ```
4. Results will be saved in the same folder:
   - `tailcor_matrix.csv`, `linear_component_matrix.csv`, `nonlinear_component_matrix.csv`
   - Clustered heatmaps (`*.png`)
   - Rolling average TailCoR plot (`avg_tailcor.png`)

