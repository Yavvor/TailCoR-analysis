import os
import zipfile
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from itertools import combinations
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================================================== CONFIG =============================================================================
MIN_REQUIRED_INSTRUMENTS = 5  # Minimum number of instruments required in a window to keep it
WINDOW_SIZE = 780  # Window size in days (publication value = 780)
STEP = 252  # Step size (publication value = 252)
ZETA = 0.95  # Publication value = 0.95
TAU = 0.75   # Publication value = 0.75

# ================================================================ LOADING AND PREPROCESSING ====================================================

def load_zip_to_df(zip_path):
    """
    Loads CSV files stored inside a ZIP archive into a single DataFrame.

    Parameters
    ----------
    zip_path : str
        Path to the ZIP archive containing CSV files.

    Returns
    -------
    DataFrame
        A concatenated DataFrame with columns ["Name", "Date", "Close"].
        - "Date" is converted to datetime.
        - Duplicates are removed.
        - Data is sorted by ["Name", "Date"].
    """
    dfs = []
    with TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            for file in files:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, sep=",", quotechar='"', engine="python")

                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
                df = df.dropna(subset=["Date", "Close", "Name"])
                df = df[["Name", "Date", "Close"]]
                df = df.sort_values(["Name", "Date"]).drop_duplicates(subset=["Name", "Date"], keep="last")
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def compute_log_returns(dfs):
    """
    Computes log returns for all instruments.

    Parameters
    ----------
    dfs : list of DataFrames
        Each DataFrame must contain ["Name", "Date", "Close"] columns.

    Returns
    -------
    DataFrame
        Pivoted DataFrame of log returns with:
        - Index = Date
        - Columns = instrument names
        - Values = log returns
    """
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort_values(by=["Name", "Date"], inplace=True)
    df_all["LogReturn"] = df_all.groupby("Name")["Close"].transform(lambda x: np.log(x).diff())
    df_all = df_all.dropna(subset=["LogReturn"])
    returns_df = df_all.pivot_table(index="Date", columns="Name", values="LogReturn", aggfunc="mean")
    return returns_df

# ============================================================== TAILCOR ===========================================================

def TailCoR(df, zeta=ZETA, tau=TAU, mode="tailcor"):
    """
    Computes TailCoR (or its components) correlation matrix for a given DataFrame of log returns.

    Parameters
    ----------
    df : DataFrame
        Index = dates, columns = instruments, values = log returns.
    zeta : float
        Parameter from publication.
    tau : float
        Parameter from publication.
    mode : {"tailcor", "linear", "nonlinear"}
        Which correlation type to compute.

    Returns
    -------
    DataFrame
        Symmetric correlation matrix of the chosen type.
    """
    cols = df.columns
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(n):
        for j in range(i, n):
            data = pd.concat([df.iloc[:, i], df.iloc[:, j]], axis=1).dropna()
            if data.shape[0] < 10:
                val = np.nan
            else:
                x_q = data.iloc[:, 0].quantile([tau, 1 - tau])
                y_q = data.iloc[:, 1].quantile([tau, 1 - tau])
                iqr_x = x_q[tau] - x_q[1 - tau]
                iqr_y = y_q[tau] - y_q[1 - tau]

                if iqr_x <= 1e-6 or iqr_y <= 1e-6:
                    val = np.nan
                else:
                    X = (data.iloc[:, 0] - data.iloc[:, 0].median()) / iqr_x
                    Y = (data.iloc[:, 1] - data.iloc[:, 1].median()) / iqr_y
                    rho = X.corr(Y)

                    if pd.isna(rho):
                        val = np.nan
                    else:
                        Z = (X + Y) / np.sqrt(2) if rho >= 0 else (X - Y) / np.sqrt(2)
                        q_upper = Z.quantile(zeta)
                        q_lower = Z.quantile(1 - zeta)
                        IQR_tail = q_upper - q_lower

                        if IQR_tail <= 1e-6:
                            val = np.nan
                        else:
                            sg = norm.ppf(tau) / norm.ppf(zeta)
                            tailcor = sg * IQR_tail
                            linear_component = np.sqrt(1 + abs(rho))
                            nonlinear_component = (
                                tailcor / linear_component
                                if linear_component > 1e-6 else np.nan
                            )

                            if mode == "tailcor":
                                val = tailcor
                            elif mode == "linear":
                                val = linear_component
                            elif mode == "nonlinear":
                                val = nonlinear_component
                            else:
                                raise ValueError("mode must be one of {'tailcor','linear','nonlinear'}")

            matrix.iat[i, j] = matrix.iat[j, i] = val

    return matrix

# ============== Pair TailCor ====================
def compute_TailCor_pairs(file_dfs, zeta=ZETA, tau=TAU):
    """
    Computes TailCoR, linear, and nonlinear matrices for each pair of instruments.

    Parameters
    ----------
    file_dfs : list of (str, DataFrame)
        List of tuples where:
        - First element = name (file or instrument identifier)
        - Second element = DataFrame with ["Name", "Date", "Close"].
    zeta : float
        Parameter from publication.
    tau : float
        Parameter from publication.

    Returns
    -------
    list of dict
        Each dict contains:
        - "zip1", "zip2" : names of the compared instruments
        - "tailcor_matrix" : TailCoR matrix
        - "linear_matrix" : linear component matrix
        - "nonlinear_matrix" : nonlinear component matrix
    """
    results = []
    all_pairs = list(combinations(file_dfs, 2))
    total_pairs = len(all_pairs)

    for idx, ((name1, df1), (name2, df2)) in enumerate(all_pairs, start=1):
        print(f"Processing pair {idx}/{total_pairs}: {name1} x {name2}")
        combined = pd.concat([df1, df2], ignore_index=True)
        returns_df = compute_log_returns([combined])

        tailcor_matrix = TailCoR(returns_df, zeta, tau, mode="tailcor")
        linear_matrix = TailCoR(returns_df, zeta, tau, mode="linear")
        nonlinear_matrix = TailCoR(returns_df, zeta, tau, mode="nonlinear")

        results.append({
            "zip1": name1,
            "zip2": name2,
            "tailcor_matrix": tailcor_matrix,
            "linear_matrix": linear_matrix,
            "nonlinear_matrix": nonlinear_matrix,
        })

    return results

# ================ Combine Matrix ===================
def build_combined_tailcor_matrices(pairwise_results):
    """
    Builds combined TailCoR, linear, and nonlinear matrices
    from pairwise results.

    Parameters
    ----------
    pairwise_results : list of dict
        Output of `compute_TailCor_pairs`.

    Returns
    -------
    tuple of DataFrames
        (combined_tailcor, combined_linear, combined_nonlinear)
        Each matrix has instruments as rows/columns.
    """
    all_names = set()
    for res in pairwise_results:
        all_names.update(res['tailcor_matrix'].columns)

    all_names = sorted(list(all_names))
    combined_tailcor = pd.DataFrame(np.nan, index=all_names, columns=all_names)
    combined_linear = pd.DataFrame(np.nan, index=all_names, columns=all_names)
    combined_nonlinear = pd.DataFrame(np.nan, index=all_names, columns=all_names)

    for res in pairwise_results:
        for matrix_name in ['tailcor_matrix', 'linear_matrix', 'nonlinear_matrix']:
            partial = res[matrix_name]
            target_matrix = {
                'tailcor_matrix': combined_tailcor,
                'linear_matrix': combined_linear,
                'nonlinear_matrix': combined_nonlinear,
            }[matrix_name]
            for i in partial.index:
                for j in partial.columns:
                    target_matrix.loc[i, j] = partial.loc[i, j]

    return combined_tailcor, combined_linear, combined_nonlinear

# ============== Average TailCor over Time =================
def avg_TailCor(df_returns, window_size=WINDOW_SIZE, step=STEP, zeta=ZETA, tau=TAU):
    """
    Computes average TailCoR, linear, and nonlinear values
    over rolling windows.

    Parameters
    ----------
    df_returns : DataFrame
        Log returns DataFrame (Date index, instruments as columns).
    window_size : int
        Number of days in each window.
    step : int
        Step size for rolling windows.
    zeta : float
        Parameter from publication.
    tau : float
        Parameter from publication.

    Returns
    -------
    DataFrame
        Time series of average TailCoR, linear, and nonlinear values.
    """
    dates = df_returns.index
    avg_tailcors = []
    total_windows = (len(dates) - window_size) // step + 1

    for window_count, start in enumerate(range(0, len(dates) - window_size + 1, step), start=1):
        end = start + window_size
        window_returns = df_returns.iloc[start:end]
        window_returns = window_returns.loc[:, window_returns.notna().sum() > 0]
        window_returns = window_returns.dropna(thresh=int(0.8 * window_returns.shape[1]))

        if window_returns.shape[1] < MIN_REQUIRED_INSTRUMENTS:
            print(f"Skipped window {window_count}/{total_windows} – too few instruments ({window_returns.shape[1]})")
            continue

        tailcor_matrix = TailCoR(window_returns, zeta, tau, mode="tailcor")
        linear_matrix = TailCoR(window_returns, zeta, tau, mode="linear")
        nonlinear_matrix = TailCoR(window_returns, zeta, tau, mode="nonlinear")

        mask = np.triu(np.ones(tailcor_matrix.shape), k=1).astype(bool)

        tailcors = tailcor_matrix.where(mask).values.flatten()
        linear_vals = linear_matrix.where(mask).values.flatten()
        nonlinear_vals = nonlinear_matrix.where(mask).values.flatten()

        tailcors = tailcors[~np.isnan(tailcors)]
        linear_vals = linear_vals[~np.isnan(linear_vals)]
        nonlinear_vals = nonlinear_vals[~np.isnan(nonlinear_vals)]

        if len(tailcors) == 0:
            print(f"Skipped window {window_count}/{total_windows} – no TailCoR values")
            continue

        avg_tailcors.append({
            "date": dates[end - 1],
            "avg_tailcor": np.mean(tailcors),
            "avg_linear": np.mean(linear_vals) if len(linear_vals) > 0 else np.nan,
            "avg_nonlinear": np.mean(nonlinear_vals) if len(nonlinear_vals) > 0 else np.nan,
        })

        print(f"Finished window {window_count}/{total_windows}")

    df_avg = pd.DataFrame(avg_tailcors)
    df_avg.set_index("date", inplace=True)
    return df_avg

# ============================================================= PLOTTING FUNCTIONS =====================================================
def plot_clustermap(matrix, title, filename, annot=True):
    """
    Plots and saves a hierarchical clustered heatmap of the given matrix.

    Parameters
    ----------
    matrix : DataFrame
        Correlation matrix (TailCoR or its components).
    title : str
        Title for the plot.
    filename : str
        Path where the figure will be saved.
    annot : bool, default=True
        Whether to annotate each cell with its value.
    """
    clean_matrix = matrix.fillna(0)
    fig_w, fig_h = 10, 10
    cbar_pos = (0.92, 0.2, 0.02, 0.55)

    g = sns.clustermap(
        clean_matrix,
        cmap="vlag",
        linewidths=0.5,
        figsize=(fig_w, fig_h),
        cbar_kws={'label': title},
        metric="euclidean",
        method="ward",
        cbar_pos=cbar_pos
    ) 
    g.fig.suptitle(title, y=0.95, fontsize=14)
    g.fig.subplots_adjust(left=0.18, right=0.80, top=0.92, bottom=0.12)
    g.cax.set_position([0.92, 0.2, 0.02, 0.55])

    if annot:
        ax = g.ax_heatmap
        for i, row in enumerate(clean_matrix.index):
            for j, col in enumerate(clean_matrix.columns):
                val = clean_matrix.loc[row, col]
                ax.text(
                    j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=6
                )

    g.fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(g.fig)

def plot_avg_tailcor(df_avg, title="Average TailCoR Evolution", filename="avg_tailcor.png"):
    """
    Plots the time evolution of average TailCoR, linear, and nonlinear values.

    Parameters
    ----------
    df_avg : DataFrame
        DataFrame with average values computed by `avg_TailCor`.
    title : str
        Title for the plot.
    filename : str
        Path where the figure will be saved.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df_avg.index, df_avg["avg_tailcor"], label="Average TailCoR")
    plt.plot(df_avg.index, df_avg["avg_linear"], label="Linear Component")
    plt.plot(df_avg.index, df_avg["avg_nonlinear"], label="Nonlinear Component")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ================================================================== EXECUTION =======================================================================

def main():
    # ============ Loading data ==========================
    folder_path = Path("/path/to/your/directory")  # Path to directory containing .zip files
    zip_files = [p for p in folder_path.glob("*.zip")]
    dfs = []

    for idx, zip_path in enumerate(zip_files, start=1):
        print(f"Loading data {idx}/{len(zip_files)}: {zip_path.name}")
        dfs.append(load_zip_to_df(zip_path))

    # ============= Calculations ============================
    returns = compute_log_returns(dfs)
    file_dfs = list(zip([p.name for p in zip_files], dfs))
    pair_results = compute_TailCor_pairs(file_dfs)
    combined_tailcor, combined_linear, combined_nonlinear = build_combined_tailcor_matrices(pair_results)

    # =============== Saving Data =======================
    combined_tailcor.to_csv(folder_path / "tailcor_matrix.csv", encoding="utf-8")
    combined_linear.to_csv(folder_path / "linear_component_matrix.csv", encoding="utf-8")
    combined_nonlinear.to_csv(folder_path / "nonlinear_component_matrix.csv", encoding="utf-8")

    # ============== Plotting ================================
    plot_clustermap(combined_tailcor, "TailCor", folder_path / "tailcor_clustermap.png")
    plot_clustermap(combined_linear, "Linear Component", folder_path / "linear_clustermap.png")
    plot_clustermap(combined_nonlinear, "Nonlinear Component", folder_path / "nonlinear_clustermap.png")

    df_avg_tailcor = avg_TailCor(returns)
    plot_avg_tailcor(df_avg_tailcor, filename=folder_path / "avg_tailcor.png")


if __name__ == "__main__":
    main()
