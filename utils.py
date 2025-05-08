"""
Utility functions for the financial network simulation.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from network_analysis import get_network_metrics_df, compare_network_metrics
from visualization import (
    plot_network, plot_centrality_histogram, plot_centrality_changes,
    plot_transaction_volume_over_time, plot_centrality_metrics_over_time,
    plot_network_structure_comparison, plot_bank_metrics_comparison,
    plot_centrality_heatmap
)


def ensure_directory(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model_data(model, filename_prefix, output_dir='output'):
    """
    Save model data to CSV files.
    
    Args:
        model: The financial network model instance
        filename_prefix: Prefix for output filenames
        output_dir: Directory to save output files
    """
    ensure_directory(output_dir)
    
    # Save model-level metrics
    model_data = model.datacollector.get_model_vars_dataframe()
    model_data.to_csv(f"{output_dir}/{filename_prefix}_model_metrics.csv")
    
    # Save agent-level metrics
    agent_data = model.datacollector.get_agent_vars_dataframe()
    agent_data.to_csv(f"{output_dir}/{filename_prefix}_agent_metrics.csv")
    
    # Save transaction history
    transactions = model.get_transaction_history_df()
    transactions.to_csv(f"{output_dir}/{filename_prefix}_transactions.csv", index=False)
    
    # Save bank interaction history
    bank_interactions = model.get_bank_interaction_history_df()
    bank_interactions.to_csv(f"{output_dir}/{filename_prefix}_bank_interactions.csv", index=False)
    
    # Save network metrics for commercial banks
    bank_metrics = get_network_metrics_df(model)
    bank_metrics.to_csv(f"{output_dir}/{filename_prefix}_bank_network_metrics.csv", index=False)
    
    print(f"Data saved to {output_dir}/ with prefix {filename_prefix}")


def generate_comparison_visualizations(model_no_cbdc, model_with_cbdc, output_dir='output'):
    """
    Generate and save comparison visualizations.
    
    Args:
        model_no_cbdc: Model instance without CBDC
        model_with_cbdc: Model instance with CBDC
        output_dir: Directory to save output files
    """
    ensure_directory(output_dir)
    
    # Get network metrics for commercial banks
    bank_metrics_no_cbdc = get_network_metrics_df(model_no_cbdc)
    bank_metrics_with_cbdc = get_network_metrics_df(model_with_cbdc)
    
    # Compare network metrics
    comparison_df = compare_network_metrics(bank_metrics_no_cbdc, bank_metrics_with_cbdc)
    comparison_df.to_csv(f"{output_dir}/metrics_comparison.csv", index=False)
    
    # Generate visualizations
    
    # 1. Network structure comparison
    fig_network = plot_network_structure_comparison(model_no_cbdc, model_with_cbdc)
    fig_network.savefig(f"{output_dir}/network_structure_comparison.png", dpi=300, bbox_inches='tight')
    
    # 2. Transaction volume over time
    fig_transactions = plot_transaction_volume_over_time(model_no_cbdc, model_with_cbdc)
    fig_transactions.savefig(f"{output_dir}/transaction_volume_comparison.png", dpi=300, bbox_inches='tight')
    
    # 3. Centrality metrics over time
    fig_centrality = plot_centrality_metrics_over_time(model_no_cbdc, model_with_cbdc)
    fig_centrality.savefig(f"{output_dir}/centrality_metrics_comparison.png", dpi=300, bbox_inches='tight')
    
    # 4. Bank metrics comparison
    fig_bank_metrics = plot_bank_metrics_comparison(bank_metrics_no_cbdc, bank_metrics_with_cbdc)
    fig_bank_metrics.savefig(f"{output_dir}/bank_metrics_comparison.png", dpi=300, bbox_inches='tight')
    
    # 5. Centrality heatmap
    fig_heatmap = plot_centrality_heatmap(comparison_df)
    fig_heatmap.savefig(f"{output_dir}/centrality_change_heatmap.png", dpi=300, bbox_inches='tight')
    
    # 6. Degree centrality changes
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_centrality_changes(comparison_df, metric='degree_centrality', ax=ax)
    fig.savefig(f"{output_dir}/degree_centrality_changes.png", dpi=300, bbox_inches='tight')
    
    # 7. Betweenness centrality changes
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_centrality_changes(comparison_df, metric='betweenness_centrality', ax=ax)
    fig.savefig(f"{output_dir}/betweenness_centrality_changes.png", dpi=300, bbox_inches='tight')
    
    # 8. Deposit changes
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_centrality_changes(comparison_df, metric='deposits', 
                           title='Changes in Bank Deposits After CBDC Introduction', ax=ax)
    fig.savefig(f"{output_dir}/deposit_changes.png", dpi=300, bbox_inches='tight')
    
    # Close all figures
    plt.close('all')
    
    print(f"Visualizations saved to {output_dir}/")


def analyze_centrality_impact(model_no_cbdc, model_with_cbdc):
    """
    Analyze the impact of CBDC on bank centrality.
    
    Args:
        model_no_cbdc: Model instance without CBDC
        model_with_cbdc: Model instance with CBDC
        
    Returns:
        dict: Dictionary with summary statistics
    """
    # Get network metrics for commercial banks
    bank_metrics_no_cbdc = get_network_metrics_df(model_no_cbdc)
    bank_metrics_with_cbdc = get_network_metrics_df(model_with_cbdc)
    
    # Compare network metrics
    comparison_df = compare_network_metrics(bank_metrics_no_cbdc, bank_metrics_with_cbdc)
    
    # Calculate summary statistics
    summary = {
        'degree_centrality': {
            'mean_before': bank_metrics_no_cbdc['degree_centrality'].mean(),
            'mean_after': bank_metrics_with_cbdc['degree_centrality'].mean(),
            'mean_pct_change': comparison_df['degree_centrality_pct_change'].mean(),
            'std_pct_change': comparison_df['degree_centrality_pct_change'].std(),
            'max_pct_change': comparison_df['degree_centrality_pct_change'].max(),
            'min_pct_change': comparison_df['degree_centrality_pct_change'].min(),
        },
        'betweenness_centrality': {
            'mean_before': bank_metrics_no_cbdc['betweenness_centrality'].mean(),
            'mean_after': bank_metrics_with_cbdc['betweenness_centrality'].mean(),
            'mean_pct_change': comparison_df['betweenness_centrality_pct_change'].mean(),
            'std_pct_change': comparison_df['betweenness_centrality_pct_change'].std(),
            'max_pct_change': comparison_df['betweenness_centrality_pct_change'].max(),
            'min_pct_change': comparison_df['betweenness_centrality_pct_change'].min(),
        },
        'eigenvector_centrality': {
            'mean_before': bank_metrics_no_cbdc['eigenvector_centrality'].mean(),
            'mean_after': bank_metrics_with_cbdc['eigenvector_centrality'].mean(),
            'mean_pct_change': comparison_df['eigenvector_centrality_pct_change'].mean(),
            'std_pct_change': comparison_df['eigenvector_centrality_pct_change'].std(),
            'max_pct_change': comparison_df['eigenvector_centrality_pct_change'].max(),
            'min_pct_change': comparison_df['eigenvector_centrality_pct_change'].min(),
        },
        'deposits': {
            'mean_before': bank_metrics_no_cbdc['deposits'].mean(),
            'mean_after': bank_metrics_with_cbdc['deposits'].mean(),
            'mean_pct_change': comparison_df['deposits_pct_change'].mean(),
            'total_before': bank_metrics_no_cbdc['deposits'].sum(),
            'total_after': bank_metrics_with_cbdc['deposits'].sum(),
            'total_pct_change': (bank_metrics_with_cbdc['deposits'].sum() - bank_metrics_no_cbdc['deposits'].sum()) / 
                               bank_metrics_no_cbdc['deposits'].sum() * 100,
        },
        'num_customers': {
            'mean_before': bank_metrics_no_cbdc['num_customers'].mean(),
            'mean_after': bank_metrics_with_cbdc['num_customers'].mean(),
            'mean_pct_change': comparison_df['num_customers_pct_change'].mean(),
            'total_before': bank_metrics_no_cbdc['num_customers'].sum(),
            'total_after': bank_metrics_with_cbdc['num_customers'].sum(),
        },
    }
    
    return summary


def print_analysis_summary(summary):
    """
    Print a summary of the analysis results.
    
    Args:
        summary: Dictionary with summary statistics
    """
    print("\n===== CBDC IMPACT ANALYSIS SUMMARY =====\n")
    
    print("NETWORK CENTRALITY CHANGES:")
    print(f"Degree Centrality: {summary['degree_centrality']['mean_pct_change']:.2f}% change")
    print(f"Betweenness Centrality: {summary['betweenness_centrality']['mean_pct_change']:.2f}% change")
    print(f"Eigenvector Centrality: {summary['eigenvector_centrality']['mean_pct_change']:.2f}% change\n")
    
    print("BANKING SYSTEM IMPACTS:")
    print(f"Total Deposits: {summary['deposits']['total_pct_change']:.2f}% change")
    print(f"Average Customers per Bank: {summary['num_customers']['mean_pct_change']:.2f}% change\n")
    
    # Interpret the results
    print("INTERPRETATION:")
    
    # Interpret degree centrality changes
    if summary['degree_centrality']['mean_pct_change'] < -5:
        print("- Significant reduction in bank interconnectedness, suggesting CBDC is displacing interbank connections")
    elif summary['degree_centrality']['mean_pct_change'] > 5:
        print("- Increased bank interconnectedness, suggesting banks are forming more connections to adapt to CBDC")
    else:
        print("- Minimal change in bank interconnectedness")
    
    # Interpret betweenness centrality changes
    if summary['betweenness_centrality']['mean_pct_change'] < -5:
        print("- Reduced bank intermediation role in the network, indicating more direct payment channels via CBDC")
    elif summary['betweenness_centrality']['mean_pct_change'] > 5:
        print("- Increased bank intermediation importance, possibly due to banks adapting to new payment flows")
    else:
        print("- Minimal change in banks' intermediation role")
    
    # Interpret deposit changes
    if summary['deposits']['total_pct_change'] < -5:
        print("- Significant deposit outflow from commercial banks to CBDC, potentially affecting bank lending capacity")
    elif summary['deposits']['total_pct_change'] > 5:
        print("- Unexpected increase in bank deposits alongside CBDC adoption")
    else:
        print("- CBDC introduction had limited impact on overall bank deposit levels")
    
    print("\n=========================================")
