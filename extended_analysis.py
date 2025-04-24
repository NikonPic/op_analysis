from pathlib import Path
from typing import Dict, List, Optional
from venv import logger

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from medical_image_analysis import CORRELATION_METHODS, ERROR_METRICS, OUTPUT_DIR, calculate_correlation_metrics, calculate_icc, create_bland_altman_plot, create_scatter_plot, extract_common_measurements
from prepare_data import OP_MODES, RELEVANT_ANGLES, RELEVANT_ANGLES_POSTOP, RELEVANT_ANGLES_PREOP


def calculate_extended_error_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    """
    Calculate extended error metrics between two sets of measurements.
    
    Args:
        actual (List[float]): Actual values (ground truth)
        predicted (List[float]): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of error metrics
    """
    if not actual or not predicted or len(actual) != len(predicted):
        return {metric: np.nan for metric in ERROR_METRICS + ['Within_1deg', 'Within_2deg', 'Within_3deg', 'Max_Error']}
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Original metrics
    me = np.mean(predicted - actual)
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    
    # Additional metrics from the second code
    # Calculate percentage within clinically acceptable thresholds
    within_1_degree = np.mean(np.abs(predicted - actual) <= 1.0) * 100
    within_2_degrees = np.mean(np.abs(predicted - actual) <= 2.0) * 100
    within_3_degrees = np.mean(np.abs(predicted - actual) <= 3.0) * 100
    
    # Maximum error
    max_error = np.max(np.abs(predicted - actual))
    
    # Median absolute error
    median_ae = np.median(np.abs(predicted - actual))
    
    return {
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "Median_AE": median_ae,
        "Within_1deg": within_1_degree,
        "Within_2deg": within_2_degrees,
        "Within_3deg": within_3_degrees,
        "Max_Error": max_error
    }

def alternative_calculate_icc(x: List[float], y: List[float]) -> float:
    """
    Calculate Intraclass Correlation Coefficient (ICC) using the method from the second code.
    
    Args:
        x (List[float]): First set of measurements
        y (List[float]): Second set of measurements
        
    Returns:
        float: ICC value
    """
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return np.nan
    
    try:
        # Create a matrix where rows are subjects and columns are raters
        ratings = np.column_stack((x, y))
        n = len(x)
        k = 2  # Number of raters
        
        # Calculate mean for each subject
        subject_means = np.mean(ratings, axis=1)
        
        # Calculate mean for each rater
        rater_means = np.mean(ratings, axis=0)
        
        # Calculate overall mean
        overall_mean = np.mean(ratings)
        
        # Calculate between-subjects sum of squares
        ss_subjects = k * np.sum((subject_means - overall_mean) ** 2)
        
        # Calculate between-raters sum of squares
        ss_raters = n * np.sum((rater_means - overall_mean) ** 2)
        
        # Calculate total sum of squares
        ss_total = np.sum((ratings - overall_mean) ** 2)
        
        # Calculate residual sum of squares
        ss_residual = ss_total - ss_subjects - ss_raters
        
        # Calculate mean squares
        ms_subjects = ss_subjects / (n - 1)
        ms_residual = ss_residual / ((n - 1) * (k - 1))
        
        # Calculate ICC (two-way mixed, absolute agreement, single rater/measurement)
        icc = (ms_subjects - ms_residual) / (ms_subjects + (k - 1) * ms_residual)
        
        return icc
    
    except Exception as e:
        logger.warning(f"Error calculating ICC with alternative method: {e}")
        return np.nan

def perform_statistical_tests(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Perform statistical tests to compare two sets of measurements.
    
    Args:
        x (List[float]): First set of measurements
        y (List[float]): Second set of measurements
        
    Returns:
        Dict[str, float]: Dictionary of test results
    """
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return {"t_test_p": np.nan, "wilcoxon_p": np.nan}
    
    try:
        # Paired t-test
        t_stat, t_p = stats.ttest_rel(x, y)
        
        # Wilcoxon signed-rank test
        w_stat, w_p = stats.wilcoxon(x, y)
        
        return {
            "t_test_p": t_p,
            "wilcoxon_p": w_p
        }
    except Exception as e:
        logger.warning(f"Error performing statistical tests: {e}")
        return {"t_test_p": np.nan, "wilcoxon_p": np.nan}

def plot_error_histogram(true_values: List[float], predicted_values: List[float], 
                        title: str, 
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create histogram of prediction errors.
    
    Args:
        true_values (List[float]): True measurements
        predicted_values (List[float]): Predicted measurements
        title (str): Plot title
        save_path (Optional[str]): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    if not true_values or not predicted_values or len(true_values) != len(predicted_values):
        logger.warning("Cannot create error histogram with empty or unequal arrays")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for error histogram", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Calculate errors
    errors = np.array(predicted_values) - np.array(true_values)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with KDE
    sns.histplot(errors, kde=True, bins=20, ax=ax)
    
    # Add vertical lines for mean and median
    ax.axvline(np.mean(errors), color='r', linestyle='--', 
              label=f'Mean: {np.mean(errors):.2f}°')
    ax.axvline(np.median(errors), color='g', linestyle='--', 
              label=f'Median: {np.median(errors):.2f}°')
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('Error (Predicted - True) (degrees)')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_error_boxplot(all_errors: List[List[float]], angle_names: List[str], 
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create boxplot comparing errors across all angles.
    
    Args:
        all_errors (List[List[float]]): List of error arrays for each angle
        angle_names (List[str]): Names of the angles
        save_path (Optional[str]): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    if not all_errors or not angle_names or len(all_errors) != len(angle_names):
        logger.warning("Cannot create error boxplot with invalid inputs")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for error boxplot", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    ax.boxplot(all_errors, labels=angle_names)
    
    # Set labels and title
    ax.set_title('Error Distribution Comparison')
    ax.set_ylabel('Error (Predicted - True) (degrees)')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_combined_error_histogram(all_errors: List[List[float]], angle_names: List[str], 
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create combined histogram of errors for all angles.
    
    Args:
        all_errors (List[List[float]]): List of error arrays for each angle
        angle_names (List[str]): Names of the angles
        save_path (Optional[str]): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    if not all_errors or not angle_names or len(all_errors) != len(angle_names):
        logger.warning("Cannot create combined error histogram with invalid inputs")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Insufficient data for combined error histogram", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a DataFrame for easier plotting
    error_df = pd.DataFrame()
    for i, errors in enumerate(all_errors):
        error_df[angle_names[i]] = pd.Series(errors)
    
    # Plot KDE for each angle
    for angle in angle_names:
        sns.kdeplot(error_df[angle], label=angle, ax=ax)
    
    # Set labels and title
    ax.set_title('Error Distribution Comparison')
    ax.set_xlabel('Error (Predicted - True) (degrees)')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation_matrix(dataset: Dict, dataset_mode: str, opmode: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create correlation matrix for all measurements in a dataset.
    
    Args:
        dataset (Dict): Dataset containing measurements
        dataset_mode (str): Dataset mode (int/ex)
        opmode (str): Operation mode
        save_path (Optional[str]): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    try:
        # Collect data for all angles
        data = {}
        display_labels = []
        
        # Extract true values for all angles (OS)
        for angle in RELEVANT_ANGLES:
            if (angle in dataset[dataset_mode][opmode]):
                values = list(dataset[dataset_mode][opmode][angle].values())
                if values:
                    data[f"{angle}_OS"] = values
                    display_labels.append(f"{angle} (OS)")
        
        # Extract predicted values for all angles (AI)
        for angle in RELEVANT_ANGLES:
            if (angle in dataset[dataset_mode][opmode]):
                values = list(dataset[dataset_mode][opmode][angle].values())
                if values:
                    data[f"{angle}_AI"] = values
                    display_labels.append(f"{angle} (AI)")
        
        if not data:
            logger.warning(f"No data available for correlation matrix: {dataset_mode}, {opmode}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning("No valid data for correlation matrix after dropping NaNs")
            return None
        
        # Calculate correlation matrix
        corr_df = df.corr()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot correlation matrix
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                   xticklabels=display_labels, yticklabels=display_labels, ax=ax)
        
        # Set title
        ax.set_title(f'Correlation Matrix - {dataset_mode.upper()}, {opmode}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return None

def plot_focused_correlation_matrix(dataset: Dict, dataset_mode: str, opmode: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create focused correlation matrix showing only OS vs AI comparisons.
    
    Args:
        dataset (Dict): Dataset containing measurements
        dataset_mode (str): Dataset mode (int/ex)
        opmode (str): Operation mode
        save_path (Optional[str]): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    try:
        # Collect data for all angles
        data = {}
        display_labels = []
        os_columns = []
        ai_columns = []
        
        # Extract true values for all angles (OS)
        for angle in RELEVANT_ANGLES:
            if (angle in dataset[dataset_mode][opmode]):
                values = list(dataset[dataset_mode][opmode][angle].values())
                if values:
                    col_name = f"{angle}_OS"
                    data[col_name] = values
                    display_labels.append(f"{angle} (OS)")
                    os_columns.append(col_name)
        
        # Extract predicted values for all angles (AI)
        for angle in RELEVANT_ANGLES:
            if (angle in dataset[dataset_mode][opmode]):
                values = list(dataset[dataset_mode][opmode][angle].values())
                if values:
                    col_name = f"{angle}_AI"
                    data[col_name] = values
                    display_labels.append(f"{angle} (AI)")
                    ai_columns.append(col_name)
        
        if not data or not os_columns or not ai_columns:
            logger.warning(f"No data available for focused correlation matrix: {dataset_mode}, {opmode}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning("No valid data for focused correlation matrix after dropping NaNs")
            return None
        
        # Calculate correlation matrix
        corr_df = df.corr()
        
        # Create mask to keep only OS vs AI correlations
        mask = np.ones_like(corr_df, dtype=bool)
        for i, col_i in enumerate(corr_df.columns):
            for j, col_j in enumerate(corr_df.columns):
                # If one is OS and one is AI
                if ((col_i in os_columns and col_j in ai_columns) or 
                    (col_i in ai_columns and col_j in os_columns)):
                    mask[i, j] = False
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot focused correlation matrix
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                   xticklabels=display_labels, yticklabels=display_labels, ax=ax)
        
        # Set title
        ax.set_title(f'Focused Correlation: OS vs AI - {dataset_mode.upper()}, {opmode}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating focused correlation matrix: {e}")
        return None

def generate_html_report(mean_os_vs_ai_results: Dict, individual_os_vs_ai_results: Dict, os_vs_os_results: Dict, output_dir: Path) -> None:
    """
    Generate comprehensive HTML report for all analyses.
    
    Args:
        mean_os_vs_ai_results (Dict): Results from mean OS vs AI analysis
        individual_os_vs_ai_results (Dict): Results from individual OS vs AI analysis
        os_vs_os_results (Dict): Results from OS vs OS analysis
        output_dir (Path): Directory to save the report
    """
    html_path = output_dir / "report.html"
    
    try:
        with open(html_path, 'w') as f:
            # HTML Header
            f.write('<html><head>')
            f.write('<style>')
            f.write('body{font-family:Arial;margin:20px;line-height:1.6}')
            f.write('table{border-collapse:collapse;width:100%;margin-bottom:20px}')
            f.write('th,td{text-align:left;padding:8px;border:1px solid #ddd}')
            f.write('th{background-color:#f2f2f2}tr:nth-child(even){background-color:#f9f9f9}')
            f.write('h1,h2,h3{color:#333}')
            f.write('.flex-container{display:flex;flex-wrap:wrap;justify-content:space-between}')
            f.write('.flex-item{flex:1;min-width:300px;margin:10px}')
            f.write('</style>')
            f.write('<title>Medical Imaging Analysis Report</title></head><body>')
            f.write('<h1>Medical Imaging Analysis Report</h1>')
            
            # Summary section
            f.write('<h2>Summary</h2>')
            f.write('<p>This report presents the results of the analysis comparing measurements between AI and orthopedic surgeons.</p>')
            
            # Mean OS vs AI Results
            f.write('<h2>Mean OS vs AI Agreement</h2>')
            
            for dataset_mode in ["int", "ex"]:
                f.write(f'<h3>{dataset_mode.upper()} Dataset</h3>')
                
                for opmode in OP_MODES:
                    f.write(f'<h4>Operation Mode: {opmode}</h4>')
                    
                    # Create summary table
                    f.write('<table>')
                    f.write('<tr><th>Angle</th><th>Samples</th><th>MAE</th><th>RMSE</th><th>ICC</th><th>Within 2°</th><th>Pearson r</th></tr>')
                    
                    for angle in RELEVANT_ANGLES:
                        if (dataset_mode in mean_os_vs_ai_results and 
                            opmode in mean_os_vs_ai_results[dataset_mode] and 
                            angle in mean_os_vs_ai_results[dataset_mode][opmode]):
                            
                            data = mean_os_vs_ai_results[dataset_mode][opmode][angle]
                            
                            # Extract metrics
                            n_samples = data.get("n_samples", 0)
                            mae = data.get("error_metrics", {}).get("MAE", np.nan)
                            rmse = data.get("error_metrics", {}).get("RMSE", np.nan)
                            icc = data.get("icc", np.nan)
                            within_2deg = data.get("error_metrics", {}).get("Within_2deg", np.nan)
                            pearson_r = data.get("correlation", {}).get("pearson", np.nan)
                            
                            # Add row to table
                            f.write(f'<tr>')
                            f.write(f'<td>{angle}</td>')
                            f.write(f'<td>{n_samples}</td>')
                            f.write(f'<td>{mae:.2f if not np.isnan(mae) else "N/A"}</td>')
                            f.write(f'<td>{rmse:.2f if not np.isnan(rmse) else "N/A"}</td>')
                            f.write(f'<td>{icc:.3f if not np.isnan(icc) else "N/A"}</td>')
                            f.write(f'<td>{within_2deg:.1f}% if not np.isnan(within_2deg) else "N/A"</td>')
                            f.write(f'<td>{pearson_r:.3f if not np.isnan(pearson_r) else "N/A"}</td>')
                            f.write(f'</tr>')
                    
                    f.write('</table>')
                    
                    # Add visualizations if they exist
                    figures_dir = output_dir / "figures" / f"{dataset_mode}_{opmode}" / "mean_os_vs_ai"
                    if figures_dir.exists():
                        f.write('<div class="flex-container">')
                        
                        for angle in RELEVANT_ANGLES:
                            scatter_path = figures_dir / f"scatter_{angle}.png"
                            bland_altman_path = figures_dir / f"bland_altman_{angle}.png"
                            error_hist_path = figures_dir / f"error_histogram_{angle}.png"
                            
                            if scatter_path.exists() or bland_altman_path.exists() or error_hist_path.exists():
                                f.write(f'<h5>{angle}</h5>')
                                f.write('<div class="flex-container">')
                                
                                if scatter_path.exists():
                                    rel_path = scatter_path.relative_to(output_dir)
                                    f.write(f'<div class="flex-item"><img src="{rel_path}" style="width:100%"/></div>')
                                
                                if bland_altman_path.exists():
                                    rel_path = bland_altman_path.relative_to(output_dir)
                                    f.write(f'<div class="flex-item"><img src="{rel_path}" style="width:100%"/></div>')
                                
                                if error_hist_path.exists():
                                    rel_path = error_hist_path.relative_to(output_dir)
                                    f.write(f'<div class="flex-item"><img src="{rel_path}" style="width:100%"/></div>')
                                
                                f.write('</div>')
                        
                        f.write('</div>')
            
            # Add summary plots
            summary_dir = output_dir / "figures" / "summary"
            if summary_dir.exists():
                f.write('<h3>Summary Plots</h3>')
                f.write('<div class="flex-container">')
                
                for dataset_mode in ["int", "ex"]:
                    icc_heatmap_path = summary_dir / f"icc_heatmap_{dataset_mode}.png"
                    icc_barplot_path = summary_dir / f"icc_barplot_{dataset_mode}.png"
                    error_boxplot_path = summary_dir / f"error_boxplot_{dataset_mode}.png"
                    combined_error_hist_path = summary_dir / f"combined_error_histogram_{dataset_mode}.png"
                    
                    if any([p.exists() for p in [icc_heatmap_path, icc_barplot_path, error_boxplot_path, combined_error_hist_path]]):
                        f.write(f'<h4>{dataset_mode.upper()} Dataset Summary</h4>')
                        f.write('<div class="flex-container">')
                        
                        for path in [icc_heatmap_path, icc_barplot_path, error_boxplot_path, combined_error_hist_path]:
                            if path.exists():
                                rel_path = path.relative_to(output_dir)
                                f.write(f'<div class="flex-item"><img src="{rel_path}" style="width:100%"/></div>')
                        
                        f.write('</div>')
                
                f.write('</div>')
            
            # Finish HTML
            f.write('</body></html>')
        
        logger.info(f"HTML report generated at {html_path}")
    
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")

def analyze_mean_os_vs_ai_extended(mean_os_data: Dict, ai_data: Dict) -> Dict:
    """
    Analyze agreement between mean OS and AI measurements with extended metrics.
    
    Args:
        mean_os_data (Dict): Mean orthopedic surgeon data
        ai_data (Dict): AI data
        
    Returns:
        Dict: Analysis results
    """
    results = {}
    
    # For storing error collections for summary plots
    all_errors = {}
    
    for dataset_mode in ["int", "ex"]:
        results[dataset_mode] = {}
        all_errors[dataset_mode] = {}
        
        for op_id, opmode in enumerate(OP_MODES):
            results[dataset_mode][opmode] = {}
            all_errors[dataset_mode][opmode] = {}
            
            cur_rel_angles = RELEVANT_ANGLES_POSTOP
            if op_id == 0:
                cur_rel_angles = RELEVANT_ANGLES_PREOP
            
            for angle in RELEVANT_ANGLES:
                # Extract common measurements
                os_values, ai_values = extract_common_measurements(
                    mean_os_data, ai_data, dataset_mode, opmode, angle
                )
                
                if not os_values or not ai_values:
                    results[dataset_mode][opmode][angle] = {
                        "n_samples": 0,
                        "error_metrics": {metric: np.nan for metric in ERROR_METRICS + ['Within_1deg', 'Within_2deg', 'Within_3deg', 'Max_Error']},
                        "correlation": {method: np.nan for method in CORRELATION_METHODS},
                        "icc": np.nan,
                        "alt_icc": np.nan,
                        "statistical_tests": {"t_test_p": np.nan, "wilcoxon_p": np.nan}
                    }
                    continue
                
                # Calculate errors for summary plots
                errors = np.array(ai_values) - np.array(os_values)
                all_errors[dataset_mode][opmode][angle] = errors.tolist()
                
                # Calculate metrics
                error_metrics = calculate_extended_error_metrics(os_values, ai_values)
                correlation = calculate_correlation_metrics(os_values, ai_values)
                icc = calculate_icc(os_values, ai_values)
                alt_icc = alternative_calculate_icc(os_values, ai_values)
                statistical_tests = perform_statistical_tests(os_values, ai_values)
                
                # Store results
                results[dataset_mode][opmode][angle] = {
                    "n_samples": len(os_values),
                    "error_metrics": error_metrics,
                    "correlation": correlation,
                    "icc": icc,
                    "alt_icc": alt_icc,
                    "statistical_tests": statistical_tests
                }
                
                # Create and save plots if we have enough data
                if len(os_values) >= 3:
                    # Directory for this specific comparison
                    plot_dir = OUTPUT_DIR / "figures" / f"{dataset_mode}_{opmode}" / "mean_os_vs_ai"
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Bland-Altman plot (already in original code)
                    title = f"Bland-Altman: Mean OS vs AI - {angle} ({dataset_mode}, {opmode})"
                    save_path = plot_dir / f"bland_altman_{angle}.png"
                    create_bland_altman_plot(
                        os_values, ai_values, title=title, save_path=str(save_path)
                    )
                    
                    # Scatter plot (already in original code)
                    title = f"Correlation: Mean OS vs AI - {angle} ({dataset_mode}, {opmode})"
                    save_path = plot_dir / f"scatter_{angle}.png"
                    create_scatter_plot(
                        os_values, ai_values, title=title, 
                        xlabel="Mean OS Measurement", ylabel="AI Measurement",
                        save_path=str(save_path)
                    )
                    
                    # Error histogram plot (new)
                    title = f"Error Distribution: Mean OS vs AI - {angle} ({dataset_mode}, {opmode})"
                    save_path = plot_dir / f"error_histogram_{angle}.png"
                    plot_error_histogram(
                        os_values, ai_values, title=title, save_path=str(save_path)
                    )
            
            # Create summary plots for this dataset mode and opmode
            if all_errors[dataset_mode][opmode]:
                # Directory for summary plots
                summary_dir = OUTPUT_DIR / "figures" / "summary"
                summary_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract errors and angle names for this combination
                all_error_arrays = []
                angle_names = []
                for angle in RELEVANT_ANGLES:
                    if angle in all_errors[dataset_mode][opmode]:
                        all_error_arrays.append(all_errors[dataset_mode][opmode][angle])
                        angle_names.append(angle)
                
                if all_error_arrays and angle_names:
                    # Error boxplot
                    save_path = summary_dir / f"error_boxplot_{dataset_mode}_{opmode}.png"
                    plot_error_boxplot(all_error_arrays, angle_names, save_path=str(save_path))
                    
                    # Combined error histogram
                    save_path = summary_dir / f"combined_error_histogram_{dataset_mode}_{opmode}.png"
                    plot_combined_error_histogram(all_error_arrays, angle_names, save_path=str(save_path))
                    
                    # Correlation matrices
                    save_path = summary_dir / f"correlation_matrix_{dataset_mode}_{opmode}.png"
                    combined_data = {
                        dataset_mode: {
                            opmode: {angle: {} for angle in RELEVANT_ANGLES}
                        }
                    }
                    
                    # Populate with OS data
                    for angle in RELEVANT_ANGLES:
                        if angle in mean_os_data[dataset_mode][opmode]:
                            combined_data[dataset_mode][opmode][angle] = mean_os_data[dataset_mode][opmode][angle]
                    
                    # Add AI data
                    for angle in RELEVANT_ANGLES:
                        if angle in ai_data[dataset_mode][opmode]:
                            combined_data[dataset_mode][opmode][f"{angle}_AI"] = ai_data[dataset_mode][opmode][angle]
                    
                    # Create correlation matrix
                    plot_correlation_matrix(combined_data, dataset_mode, opmode, save_path=str(save_path))
                    
                    # Create focused correlation matrix
                    save_path = summary_dir / f"focused_correlation_matrix_{dataset_mode}_{opmode}.png"
                    plot_focused_correlation_matrix(combined_data, dataset_mode, opmode, save_path=str(save_path))