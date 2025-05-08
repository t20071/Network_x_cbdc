"""
CBDC (Central Bank Digital Currency) Impact Simulation
Main script to run the financial network simulation and analyze results.
"""
import os
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from model import FinancialNetworkModel
from utils import (
    save_model_data,
    generate_comparison_visualizations,
    analyze_centrality_impact,
    print_analysis_summary
)


def run_simulation(steps=100, cbdc_active=False, random_seed=None):
    """
    Run a single simulation with or without CBDC.
    
    Args:
        steps: Number of simulation steps to run
        cbdc_active: Whether CBDC is active in the simulation
        random_seed: Random seed for reproducibility
        
    Returns:
        The completed model instance
    """
    print(f"Running simulation with CBDC {'active' if cbdc_active else 'inactive'} for {steps} steps...")
    
    # Create model
    model = FinancialNetworkModel(
        num_commercial_banks=20,
        num_individuals=1000,
        merchant_ratio=0.1,
        cbdc_active=cbdc_active,
        random_seed=random_seed
    )
    
    # Run model for specified number of steps
    for i in range(steps):
        model.step()
        if (i+1) % 10 == 0:
            print(f"  Step {i+1}/{steps} completed")
    
    print("Simulation complete!")
    return model


def main():
    """Main function to run the simulations and generate analysis"""
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run simulations
    print("\n1. Running simulations...")
    # First simulation without CBDC
    model_no_cbdc = run_simulation(steps=100, cbdc_active=False, random_seed=random_seed)
    
    # Second simulation with CBDC
    model_with_cbdc = run_simulation(steps=100, cbdc_active=True, random_seed=random_seed)
    
    # Save simulation data to CSV files
    print("\n2. Saving simulation data...")
    save_model_data(model_no_cbdc, "no_cbdc", output_dir)
    save_model_data(model_with_cbdc, "with_cbdc", output_dir)
    
    # Generate and save visualizations
    print("\n3. Generating visualizations...")
    generate_comparison_visualizations(model_no_cbdc, model_with_cbdc, output_dir)
    
    # Analyze the impact of CBDC on network centrality
    print("\n4. Analyzing CBDC impact...")
    summary = analyze_centrality_impact(model_no_cbdc, model_with_cbdc)
    
    # Save summary to CSV
    summary_df = pd.DataFrame({
        'Metric': [
            'Degree Centrality Mean Before',
            'Degree Centrality Mean After',
            'Degree Centrality Mean % Change',
            'Betweenness Centrality Mean Before',
            'Betweenness Centrality Mean After',
            'Betweenness Centrality Mean % Change',
            'Eigenvector Centrality Mean Before',
            'Eigenvector Centrality Mean After',
            'Eigenvector Centrality Mean % Change',
            'Deposits Mean Before',
            'Deposits Mean After',
            'Deposits Mean % Change',
            'Deposits Total Before',
            'Deposits Total After',
            'Deposits Total % Change',
            'Customers Mean Before',
            'Customers Mean After',
            'Customers Mean % Change',
        ],
        'Value': [
            summary['degree_centrality']['mean_before'],
            summary['degree_centrality']['mean_after'],
            summary['degree_centrality']['mean_pct_change'],
            summary['betweenness_centrality']['mean_before'],
            summary['betweenness_centrality']['mean_after'],
            summary['betweenness_centrality']['mean_pct_change'],
            summary['eigenvector_centrality']['mean_before'],
            summary['eigenvector_centrality']['mean_after'],
            summary['eigenvector_centrality']['mean_pct_change'],
            summary['deposits']['mean_before'],
            summary['deposits']['mean_after'],
            summary['deposits']['mean_pct_change'],
            summary['deposits']['total_before'],
            summary['deposits']['total_after'],
            summary['deposits']['total_pct_change'],
            summary['num_customers']['mean_before'],
            summary['num_customers']['mean_after'],
            summary['num_customers']['mean_pct_change'],
        ]
    })
    summary_df.to_csv(f"{output_dir}/impact_summary.csv", index=False)
    
    # Print summary
    print_analysis_summary(summary)
    
    print(f"\nSimulation complete! All results saved to {output_dir}/ directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
