# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Imaging Data Analysis Script - Analysis Module
Performs comprehensive comparison between AI and orthopedic surgeon measurements
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import logging
from pathlib import Path
import statsmodels.api as sm
from itertools import combinations

# Import the data loading functions from your previous script
# Assuming the previous script is saved as data_loader.py
from prepare_data import load_surgeon_data, load_ai_data, SURGEON_SHEET_NAMES, OP_MODES, RELEVANT_ANGLES, RELEVANT_ANGLES_PREOP, RELEVANT_ANGLES_POSTOP

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
(OUTPUT_DIR / "stats").mkdir(exist_ok=True)

# Analysis settings
BLAND_ALTMAN_CONFIDENCE = 0.95  # 95% confidence interval
CORRELATION_METHODS = ["pearson", "spearman"]
ERROR_METRICS = ["MAE", "RMSE", "ME"]  # Mean Absolute Error, Root Mean Square Error, Mean Error

def calculate_mean_os_data(os_data: Dict) -> Dict:
    """
    Calculate mean values across all surgeons for each measurement.
    Handles missing, NaN, and invalid values by excluding them from mean calculation.
   
    Args:
        os_data (Dict): Orthopedic surgeon data
       
    Returns:
        Dict: Mean orthopedic surgeon data
    """
    mean_os_data = {}
    logger.info("Calculating mean orthopedic surgeon data")
   
    for dataset_mode in ["int", "ex"]:
        mean_os_data[dataset_mode] = {}
       
        for opmode in OP_MODES:
            mean_os_data[dataset_mode][opmode] = {}
           
            for angle in RELEVANT_ANGLES:
                mean_os_data[dataset_mode][opmode][angle] = {}
               
                try:
                    # Collect all patient_ids across all surgeons for this combination
                    all_patient_ids = set()
                    for sheet_name in SURGEON_SHEET_NAMES:
                        try:
                            if (sheet_name in os_data and 
                                dataset_mode in os_data[sheet_name] and 
                                opmode in os_data[sheet_name][dataset_mode] and 
                                angle in os_data[sheet_name][dataset_mode][opmode]):
                                all_patient_ids.update(os_data[sheet_name][dataset_mode][opmode][angle].keys())
                        except Exception as e:
                            logger.warning(f"Error collecting patient IDs for {sheet_name}, {dataset_mode}, {opmode}, {angle}: {e}")
                            continue
                    
                    if not all_patient_ids:
                        logger.info(f"No patient IDs found for {dataset_mode}, {opmode}, {angle}")
                        continue
                   
                    # For each patient, calculate mean across surgeons
                    for patient_id in all_patient_ids:
                        valid_values = []
                        for sheet_name in SURGEON_SHEET_NAMES:
                            try:
                                if (sheet_name in os_data and 
                                    dataset_mode in os_data[sheet_name] and 
                                    opmode in os_data[sheet_name][dataset_mode] and 
                                    angle in os_data[sheet_name][dataset_mode][opmode]):
                                    
                                    value = os_data[sheet_name][dataset_mode][opmode][angle].get(patient_id)
                                    
                                    # Only include valid numerical values (not None, not NaN)
                                    if value is not None and not pd.isna(value):
                                        # Check if value is numerical
                                        try:
                                            float_value = float(value)
                                            valid_values.append(float_value)
                                        except (ValueError, TypeError):
                                            logger.warning(f"Non-numerical value for {patient_id}, {sheet_name}, {angle}: {value}")
                            except Exception as e:
                                logger.warning(f"Error retrieving value for {patient_id}, {sheet_name}, {dataset_mode}, {opmode}, {angle}: {e}")
                                continue
                       
                        # Only calculate mean if we have at least one valid value
                        if valid_values:
                            mean_os_data[dataset_mode][opmode][angle][patient_id] = np.mean(valid_values)
                            
                        if len(valid_values) < len(SURGEON_SHEET_NAMES):
                            logger.info(f"Patient {patient_id}, {angle}: Only {len(valid_values)} out of {len(SURGEON_SHEET_NAMES)} surgeons have valid measurements")
                
                except Exception as e:
                    logger.error(f"Error calculating mean for {dataset_mode}, {opmode}, {angle}: {e}")
    
    return mean_os_data

def extract_common_measurements(data1: Dict, data2: Dict, 
                               dataset_mode: str, opmode: str, angle: str) -> Tuple[List, List]:
    """
    Extract paired measurements where both sources have data.
    
    Args:
        data1 (Dict): First data source
        data2 (Dict): Second data source
        dataset_mode (str): Dataset mode (int/ex)
        opmode (str): Operation mode
        angle (str): Angle measurement
        
    Returns:
        Tuple[List, List]: Paired measurements from both sources
    """
    values1 = []
    values2 = []
    
    try:
        data1_patients = data1[dataset_mode][opmode][angle]
        data2_patients = data2[dataset_mode][opmode][angle]
        
        # Find common patient_ids
        common_patients = set(data1_patients.keys()) & set(data2_patients.keys())
        
        for patient_id in common_patients:
            values1.append(data1_patients[patient_id])
            values2.append(data2_patients[patient_id])
            
        return values1, values2
    
    except KeyError as e:
        logger.warning(f"Key error when extracting common measurements: {e}")
        return [], []

def calculate_error_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    """
    Calculate error metrics between two sets of measurements.
    
    Args:
        actual (List[float]): Actual values (ground truth)
        predicted (List[float]): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of error metrics
    """
    if not actual or not predicted or len(actual) != len(predicted):
        return {metric: np.nan for metric in ERROR_METRICS}
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Mean Error (ME) - indicates bias direction
    me = np.mean(predicted - actual)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted - actual))
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    
    return {
        "ME": me,
        "MAE": mae,
        "RMSE": rmse
    }

def calculate_correlation_metrics(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate correlation metrics between two sets of measurements.
    
    Args:
        x (List[float]): First set of measurements
        y (List[float]): Second set of measurements
        
    Returns:
        Dict[str, float]: Dictionary of correlation metrics
    """
    if not x or not y or len(x) != len(y):
        return {method: np.nan for method in CORRELATION_METHODS}
    
    results = {}
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    results["pearson"] = pearson_r
    results["pearson_p"] = pearson_p
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x, y)
    results["spearman"] = spearman_r
    results["spearman_p"] = spearman_p
    
    return results

def calculate_icc(x: List[float], y: List[float]) -> float:
    """
    Calculate Intraclass Correlation Coefficient (ICC).
    
    Args:
        x (List[float]): First set of measurements
        y (List[float]): Second set of measurements
        
    Returns:
        float: ICC value
    """
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return np.nan
    
    try:
        # Create a dataframe with data in long format
        data = []
        for i in range(len(x)):
            data.append({"subject": i, "rater": 1, "measurement": x[i]})
            data.append({"subject": i, "rater": 2, "measurement": y[i]})
        
        df = pd.DataFrame(data)
        
        # Fit a mixed effects model
        formula = "measurement ~ 1"
        model = sm.MixedLM.from_formula(formula, groups="subject", data=df)
        result = model.fit()
        
        # Calculate ICC based on the model
        subj_var = result.cov_re.iloc[0, 0]
        resid_var = result.scale
        icc = subj_var / (subj_var + resid_var)
        
        return icc
    
    except Exception as e:
        logger.warning(f"Error calculating ICC: {e}")
        return np.nan

def create_bland_altman_plot(x: List[float], y: List[float], 
                           title: str = "", 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a Bland-Altman plot to assess agreement between two methods.
    
    Args:
        x (List[float]): Measurements from method 1
        y (List[float]): Measurements from method 2
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    if not x or not y or len(x) != len(y):
        logger.warning("Cannot create Bland-Altman plot with empty or unequal arrays")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for Bland-Altman plot", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate mean and difference
    mean = np.mean([x, y], axis=0)
    diff = y - x  # Method 2 - Method 1
    
    # Calculate mean difference and standard deviation
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    # Calculate limits of agreement
    upper_loa = md + 1.96 * sd
    lower_loa = md - 1.96 * sd
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    ax.scatter(mean, diff, alpha=0.7)
    
    # Add horizontal lines for mean difference and limits of agreement
    ax.axhline(md, color='k', linestyle='-', linewidth=1)
    ax.axhline(upper_loa, color='r', linestyle='--', linewidth=1)
    ax.axhline(lower_loa, color='r', linestyle='--', linewidth=1)
    
    # Add text annotations
    ax.text(np.max(mean), md, f'Mean diff: {md:.2f}', 
            horizontalalignment='right', verticalalignment='bottom')
    ax.text(np.max(mean), upper_loa, f'Upper LoA: {upper_loa:.2f}', 
            horizontalalignment='right', verticalalignment='bottom')
    ax.text(np.max(mean), lower_loa, f'Lower LoA: {lower_loa:.2f}', 
            horizontalalignment='right', verticalalignment='top')
    
    # Set labels and title
    ax.set_xlabel('Mean of Methods')
    ax.set_ylabel('Difference between Methods (Method 2 - Method 1)')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_scatter_plot(x: List[float], y: List[float], 
                      title: str = "", 
                      xlabel: str = "Method 1", 
                      ylabel: str = "Method 2",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a scatter plot with correlation metrics.
    
    Args:
        x (List[float]): Measurements from method 1
        y (List[float]): Measurements from method 2
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created figure
    """
    if not x or not y or len(x) != len(y):
        logger.warning("Cannot create scatter plot with empty or unequal arrays")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for scatter plot", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Calculate correlation metrics
    corr_metrics = calculate_correlation_metrics(x, y)
    pearson_r = corr_metrics.get("pearson", np.nan)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    ax.scatter(x, y, alpha=0.7)
    
    # Add regression line
    if not np.isnan(pearson_r) and len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, 'r-', linewidth=1)
    
    # Add identity line (y=x)
    ax.plot([min(x), max(x)], [min(x), max(x)], 'k--', alpha=0.5)
    
    # Add correlation info
    if not np.isnan(pearson_r):
        text = f'Pearson r: {pearson_r:.3f}\n'
        text += f'Spearman r: {corr_metrics.get("spearman", np.nan):.3f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_mean_os_vs_ai(mean_os_data: Dict, ai_data: Dict) -> Dict:
    """
    Analyze agreement between mean OS and AI measurements.
    
    Args:
        mean_os_data (Dict): Mean orthopedic surgeon data
        ai_data (Dict): AI data
        
    Returns:
        Dict: Analysis results
    """
    results = {}
    
    for dataset_mode in ["int", "ex"]:
        results[dataset_mode] = {}
        
        for op_id, opmode in enumerate(OP_MODES):
            print(opmode)
            results[dataset_mode][opmode] = {}
            
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
                        "error_metrics": {metric: np.nan for metric in ERROR_METRICS},
                        "correlation": {method: np.nan for method in CORRELATION_METHODS},
                        "icc": np.nan
                    }
                    continue
                
                # Calculate metrics
                error_metrics = calculate_error_metrics(os_values, ai_values)
                correlation = calculate_correlation_metrics(os_values, ai_values)
                icc = calculate_icc(os_values, ai_values)
                
                # Store results
                results[dataset_mode][opmode][angle] = {
                    "n_samples": len(os_values),
                    "error_metrics": error_metrics,
                    "correlation": correlation,
                    "icc": icc
                }
                
                # Create and save plots if we have enough data
                if len(os_values) >= 3:
                    # Directory for this specific comparison
                    plot_dir = OUTPUT_DIR / "figures" / f"{dataset_mode}_{opmode}" / "mean_os_vs_ai"
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Bland-Altman plot
                    title = f"Bland-Altman: Mean OS vs AI - {angle} ({dataset_mode}, {opmode})"
                    save_path = plot_dir / f"bland_altman_{angle}.png"
                    create_bland_altman_plot(
                        os_values, ai_values, title=title, save_path=str(save_path)
                    )
                    
                    # Scatter plot
                    title = f"Correlation: Mean OS vs AI - {angle} ({dataset_mode}, {opmode})"
                    save_path = plot_dir / f"scatter_{angle}.png"
                    create_scatter_plot(
                        os_values, ai_values, title=title, 
                        xlabel="Mean OS Measurement", ylabel="AI Measurement",
                        save_path=str(save_path)
                    )
    
    return results

def analyze_individual_os_vs_ai(os_data: Dict, ai_data: Dict) -> Dict:
    """
    Analyze agreement between individual OS and AI measurements.
    
    Args:
        os_data (Dict): Orthopedic surgeon data
        ai_data (Dict): AI data
        
    Returns:
        Dict: Analysis results
    """
    results = {}
    
    for sheet_name in SURGEON_SHEET_NAMES:
        results[sheet_name] = {}
        
        for dataset_mode in ["int", "ex"]:
            results[sheet_name][dataset_mode] = {}
            
            for opmode in OP_MODES:
                results[sheet_name][dataset_mode][opmode] = {}
                
                for angle in RELEVANT_ANGLES:
                    # Extract the surgeon's data for this combination
                    try:
                        if sheet_name not in os_data or \
                           dataset_mode not in os_data[sheet_name] or \
                           opmode not in os_data[sheet_name][dataset_mode] or \
                           angle not in os_data[sheet_name][dataset_mode][opmode]:
                            # Skip if data is missing at any level
                            continue
                        
                        surgeon_data = {
                            dataset_mode: {
                                opmode: {
                                    angle: os_data[sheet_name][dataset_mode][opmode][angle]
                                }
                            }
                        }
                        
                        # Extract common measurements
                        os_values, ai_values = extract_common_measurements(
                            surgeon_data, ai_data, dataset_mode, opmode, angle
                        )
                        
                        if not os_values or not ai_values:
                            results[sheet_name][dataset_mode][opmode][angle] = {
                                "n_samples": 0,
                                "error_metrics": {metric: np.nan for metric in ERROR_METRICS},
                                "correlation": {method: np.nan for method in CORRELATION_METHODS},
                                "icc": np.nan
                            }
                            continue
                        
                        # Calculate metrics
                        error_metrics = calculate_error_metrics(os_values, ai_values)
                        correlation = calculate_correlation_metrics(os_values, ai_values)
                        icc = calculate_icc(os_values, ai_values)
                        
                        # Store results
                        results[sheet_name][dataset_mode][opmode][angle] = {
                            "n_samples": len(os_values),
                            "error_metrics": error_metrics,
                            "correlation": correlation,
                            "icc": icc
                        }
                        
                        # Create and save plots if we have enough data
                        if len(os_values) >= 3:
                            # Directory for this specific comparison
                            plot_dir = OUTPUT_DIR / "figures" / f"{dataset_mode}_{opmode}" / f"{sheet_name}_vs_ai"
                            plot_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Bland-Altman plot
                            title = f"Bland-Altman: {sheet_name} vs AI - {angle} ({dataset_mode}, {opmode})"
                            save_path = plot_dir / f"bland_altman_{angle}.png"
                            create_bland_altman_plot(
                                os_values, ai_values, title=title, save_path=str(save_path)
                            )
                            
                            # Scatter plot
                            title = f"Correlation: {sheet_name} vs AI - {angle} ({dataset_mode}, {opmode})"
                            save_path = plot_dir / f"scatter_{angle}.png"
                            create_scatter_plot(
                                os_values, ai_values, title=title, 
                                xlabel=f"{sheet_name} Measurement", ylabel="AI Measurement",
                                save_path=str(save_path)
                            )
                    
                    except Exception as e:
                        logger.error(f"Error in individual OS vs AI analysis for {sheet_name}, "
                                    f"{dataset_mode}, {opmode}, {angle}: {e}")
    
    return results

def analyze_os_vs_os(os_data: Dict) -> Dict:
    """
    Analyze inter-observer agreement between orthopedic surgeons.
    
    Args:
        os_data (Dict): Orthopedic surgeon data
        
    Returns:
        Dict: Analysis results
    """
    results = {}
    
    # Get all pairs of surgeons
    surgeon_pairs = list(combinations(SURGEON_SHEET_NAMES, 2))
    
    for dataset_mode in ["int", "ex"]:
        results[dataset_mode] = {}
        
        for opmode in OP_MODES:
            results[dataset_mode][opmode] = {}
            
            for angle in RELEVANT_ANGLES:
                results[dataset_mode][opmode][angle] = {}
                
                # Analyze each pair of surgeons
                for surgeon1, surgeon2 in surgeon_pairs:
                    pair_name = f"{surgeon1}_vs_{surgeon2}"
                    
                    # Extract individual surgeon data for this combination
                    try:
                        surgeon1_data = {
                            dataset_mode: {
                                opmode: {
                                    angle: os_data[surgeon1][dataset_mode][opmode][angle]
                                }
                            }
                        }
                        
                        surgeon2_data = {
                            dataset_mode: {
                                opmode: {
                                    angle: os_data[surgeon2][dataset_mode][opmode][angle]
                                }
                            }
                        }
                        
                        # Extract common measurements
                        surgeon1_values, surgeon2_values = extract_common_measurements(
                            surgeon1_data, surgeon2_data, dataset_mode, opmode, angle
                        )
                        
                        if not surgeon1_values or not surgeon2_values:
                            results[dataset_mode][opmode][angle][pair_name] = {
                                "n_samples": 0,
                                "error_metrics": {metric: np.nan for metric in ERROR_METRICS},
                                "correlation": {method: np.nan for method in CORRELATION_METHODS},
                                "icc": np.nan
                            }
                            continue
                        
                        # Calculate metrics
                        error_metrics = calculate_error_metrics(surgeon1_values, surgeon2_values)
                        correlation = calculate_correlation_metrics(surgeon1_values, surgeon2_values)
                        icc = calculate_icc(surgeon1_values, surgeon2_values)
                        
                        # Store results
                        results[dataset_mode][opmode][angle][pair_name] = {
                            "n_samples": len(surgeon1_values),
                            "error_metrics": error_metrics,
                            "correlation": correlation,
                            "icc": icc
                        }
                        
                        # Create and save plots if we have enough data
                        if len(surgeon1_values) >= 3:
                            # Directory for this specific comparison
                            plot_dir = OUTPUT_DIR / "figures" / f"{dataset_mode}_{opmode}" / "os_vs_os"
                            plot_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Bland-Altman plot
                            title = f"Bland-Altman: {surgeon1} vs {surgeon2} - {angle} ({dataset_mode}, {opmode})"
                            save_path = plot_dir / f"bland_altman_{pair_name}_{angle}.png"
                            create_bland_altman_plot(
                                surgeon1_values, surgeon2_values, title=title, save_path=str(save_path)
                            )
                            
                            # Scatter plot
                            title = f"Correlation: {surgeon1} vs {surgeon2} - {angle} ({dataset_mode}, {opmode})"
                            save_path = plot_dir / f"scatter_{pair_name}_{angle}.png"
                            create_scatter_plot(
                                surgeon1_values, surgeon2_values, title=title, 
                                xlabel=f"{surgeon1} Measurement", ylabel=f"{surgeon2} Measurement",
                                save_path=str(save_path)
                            )
                    
                    except Exception as e:
                        logger.error(f"Error in OS vs OS analysis for {pair_name}, "
                                    f"{dataset_mode}, {opmode}, {angle}: {e}")
    
    return results

def export_results_to_csv(results: Dict, filename: str) -> None:
    """
    Export analysis results to CSV file.
    
    Args:
        results (Dict): Analysis results
        filename (str): Output filename
    """
    output_path = OUTPUT_DIR / "tables" / filename
    
    # Flatten nested dictionary into rows for CSV
    rows = []
    
    # Handle different result structures
    if "int" in results.keys():  # Mean OS vs AI or OS vs OS structure
        for dataset_mode in ["int", "ex"]:
            for opmode in OP_MODES:
                for angle in RELEVANT_ANGLES:
                    if dataset_mode in results and opmode in results[dataset_mode] and angle in results[dataset_mode][opmode]:
                        data = results[dataset_mode][opmode][angle]
                        
                        # Handle OS vs OS structure (has pairs)
                        if isinstance(data, dict) and (any(key.endswith("_vs_") for key in data.keys()) or any("_vs_" in key for key in data.keys())):
                            for pair_name, pair_data in data.items():
                                row = {
                                    "Dataset": dataset_mode,
                                    "Operation Mode": opmode,
                                    "Angle": angle,
                                    "Comparison": pair_name,
                                    "n_samples": pair_data.get("n_samples", 0)
                                }
                                
                                # Add error metrics
                                for metric, value in pair_data.get("error_metrics", {}).items():
                                    row[metric] = value
                                
                                # Add correlation metrics
                                for method, value in pair_data.get("correlation", {}).items():
                                    row[f"{method}_r"] = value
                                
                                # Add ICC
                                row["ICC"] = pair_data.get("icc", np.nan)
                                
                                rows.append(row)
                        else:
                            # Standard structure (mean OS vs AI)
                            row = {
                                "Dataset": dataset_mode,
                                "Operation Mode": opmode,
                                "Angle": angle,
                                "n_samples": data.get("n_samples", 0)
                            }
                            
                            # Add error metrics
                            for metric, value in data.get("error_metrics", {}).items():
                                row[metric] = value
                            
                            # Add correlation metrics
                            for method, value in data.get("correlation", {}).items():
                                row[f"{method}_r"] = value
                            
                            # Add ICC
                            row["ICC"] = data.get("icc", np.nan)
                            
                            rows.append(row)
    else:
        for sheet_name in results.keys():  # Individual OS vs AI structure
            for dataset_mode in ["int", "ex"]:
                for opmode in OP_MODES:
                    for angle in RELEVANT_ANGLES:
                        if (sheet_name in results and 
                            dataset_mode in results[sheet_name] and 
                            opmode in results[sheet_name][dataset_mode] and 
                            angle in results[sheet_name][dataset_mode][opmode]):
                            
                            data = results[sheet_name][dataset_mode][opmode][angle]
                            
                            row = {
                                "Surgeon": sheet_name,
                                "Dataset": dataset_mode,
                                "Operation Mode": opmode,
                                "Angle": angle,
                                "n_samples": data.get("n_samples", 0)
                            }
                            
                            # Add error metrics
                            for metric, value in data.get("error_metrics", {}).items():
                                row[metric] = value
                            
                            # Add correlation metrics
                            for method, value in data.get("correlation", {}).items():
                                row[f"{method}_r"] = value
                            
                            # Add ICC
                            row["ICC"] = data.get("icc", np.nan)
                            
                            rows.append(row)
    
    # Create DataFrame and save to CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        logger.info(f"Results exported to {output_path}")
    else:
        logger.warning(f"No results to export to {output_path}")
                            
                            
def create_summary_plots(mean_os_vs_ai_results: Dict) -> None:
    """
    Create summary plots for the entire analysis.
    
    Args:
        mean_os_vs_ai_results (Dict): Results from mean OS vs AI analysis
    """
    # Create ICC summary plot for each dataset mode
    for dataset_mode in ["int", "ex"]:
        # Prepare data for plotting
        angles = []
        opmodes = []
        icc_values = []
        
        for opmode in OP_MODES:
            for angle in RELEVANT_ANGLES:
                if (dataset_mode in mean_os_vs_ai_results and 
                    opmode in mean_os_vs_ai_results[dataset_mode] and 
                    angle in mean_os_vs_ai_results[dataset_mode][opmode]):
                    
                    data = mean_os_vs_ai_results[dataset_mode][opmode][angle]
                    icc = data.get("icc", np.nan)
                    
                    if not np.isnan(icc):
                        angles.append(angle)
                        opmodes.append(opmode)
                        icc_values.append(icc)
        
        if not angles:
            continue  # Skip if no data
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Angle": angles,
            "Operation Mode": opmodes,
            "ICC": icc_values
        })
        
        # Create directory for summary plots
        summary_dir = OUTPUT_DIR / "figures" / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ICC heatmap
        plt.figure(figsize=(12, 8))
        pivot_table = df.pivot_table(values="ICC", index="Angle", columns="Operation Mode")
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
        plt.title(f"ICC: Mean OS vs AI - {dataset_mode.upper()} Dataset")
        plt.tight_layout()
        plt.savefig(summary_dir / f"icc_heatmap_{dataset_mode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create ICC barplot
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Angle", y="ICC", hue="Operation Mode", data=df)
        plt.title(f"ICC: Mean OS vs AI - {dataset_mode.upper()} Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(summary_dir / f"icc_barplot_{dataset_mode}.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Main function to run the entire analysis pipeline.
    """
    logger.info("Starting medical imaging analysis")
    
    # Load data
    logger.info("Loading surgeon data")
    os_data, unique_ids = load_surgeon_data()
    
    logger.info("Loading AI data")
    ai_data = load_ai_data(unique_ids)
    
    # Calculate mean OS data
    logger.info("Calculating mean orthopedic surgeon measurements")
    mean_os_data = calculate_mean_os_data(os_data)
    
    # Run analyses
    logger.info("Analyzing mean OS vs AI agreement")
    mean_os_vs_ai_results = analyze_mean_os_vs_ai(mean_os_data, ai_data)
    
    logger.info("Analyzing individual OS vs AI agreement")
    #individual_os_vs_ai_results = analyze_individual_os_vs_ai(os_data, ai_data)
    
    logger.info("Analyzing inter-observer agreement between surgeons")
    #os_vs_os_results = analyze_os_vs_os(os_data)
    
    # Export results
    logger.info("Exporting results to CSV")
    export_results_to_csv(mean_os_vs_ai_results, "mean_os_vs_ai_results.csv")
    #export_results_to_csv(individual_os_vs_ai_results, "individual_os_vs_ai_results.csv")
    #export_results_to_csv(os_vs_os_results, "os_vs_os_results.csv")
    
    # Create summary plots
    logger.info("Creating summary plots")
    create_summary_plots(mean_os_vs_ai_results)
    
    # Create formatted summary tables
    logger.info("Creating formatted summary tables")
    #create_summary_tables(mean_os_vs_ai_results, individual_os_vs_ai_results, os_vs_os_results)
    
    #logger.info("Analysis complete!")
    return mean_os_vs_ai_results

if __name__ == "__main__":
    mean_os_vs_ai_results = main()
