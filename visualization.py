"""
Visualization functions for the financial network simulation.
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_network(model, title="Financial Network", bank_cmap='Blues', individual_color='lightgrey', 
                central_bank_color='red', size_factor=100, ax=None):
    """
    Plot the financial network.
    
    Args:
        model: The financial network model instance
        title: Title for the plot
        bank_cmap: Colormap for commercial banks (varies by centrality)
        individual_color: Color for individual agents
        central_bank_color: Color for central bank
        size_factor: Factor to scale node sizes
        ax: Matplotlib axis to plot on
    """
    G = model.G
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get positions for nodes
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Get agent types and centrality values
    node_types = {}
    node_colors = []
    node_sizes = []
    
    for agent in model.schedule.agents:
        node_types[agent.unique_id] = agent.type
        
        # Determine node color based on type
        if agent.type == "central_bank":
            node_colors.append(central_bank_color)
            node_sizes.append(300)  # Larger size for central bank
        elif agent.type == "commercial_bank":
            # Use centrality for color intensity
            centrality = getattr(agent, 'degree_centrality', 0)
            cmap = plt.cm.get_cmap(bank_cmap)
            node_colors.append(cmap(min(centrality * 3, 0.9)))  # Scale centrality for better color distribution
            node_sizes.append(150 + agent.capital / 100)  # Size based on capital
        else:  # Individual
            node_colors.append(individual_color)
            node_sizes.append(20)  # Smaller size for individuals
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    
    # Draw edges with varying width based on weight
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.1 + (w / max_weight) * 2 for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='grey', arrows=False, ax=ax)
    
    # Add labels for banks only (to avoid cluttering)
    bank_labels = {n: str(n) for n in G.nodes() if node_types.get(n) in ["central_bank", "commercial_bank"]}
    nx.draw_networkx_labels(G, pos, labels=bank_labels, font_size=8, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def plot_centrality_histogram(bank_metrics_df, centrality_type='degree_centrality', 
                             title=None, bins=10, ax=None):
    """
    Plot histogram of bank centrality metrics.
    
    Args:
        bank_metrics_df: DataFrame with bank metrics
        centrality_type: Type of centrality to plot
        title: Plot title
        bins: Number of histogram bins
        ax: Matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if title is None:
        title = f'Distribution of Bank {centrality_type.replace("_", " ").title()}'
    
    ax.hist(bank_metrics_df[centrality_type], bins=bins, alpha=0.7, color='steelblue')
    ax.set_xlabel(centrality_type.replace("_", " ").title())
    ax.set_ylabel('Number of Banks')
    ax.set_title(title)
    
    # Add mean line
    mean_value = bank_metrics_df[centrality_type].mean()
    ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    ax.text(mean_value * 1.1, ax.get_ylim()[1] * 0.9, f'Mean: {mean_value:.4f}', 
            color='red', fontsize=10)
    
    return ax


def plot_centrality_changes(comparison_df, metric='degree_centrality', 
                           title=None, ax=None):
    """
    Plot changes in centrality metrics before and after CBDC.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        metric: Centrality metric to plot
        title: Plot title
        ax: Matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is None:
        title = f'Changes in Bank {metric.replace("_", " ").title()} After CBDC Introduction'
    
    # Sort by before value
    sorted_df = comparison_df.sort_values(f'{metric}_before')
    
    # Plot before and after values
    x = np.arange(len(sorted_df))
    width = 0.35
    
    ax.bar(x - width/2, sorted_df[f'{metric}_before'], width, label='Before CBDC', color='steelblue')
    ax.bar(x + width/2, sorted_df[f'{metric}_after'], width, label='After CBDC', color='darkorange')
    
    ax.set_xlabel('Bank ID')
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['bank_id'], rotation=90)
    ax.legend()
    
    plt.tight_layout()
    return ax


def plot_transaction_volume_over_time(model_no_cbdc, model_with_cbdc, window=5):
    """
    Plot transaction volumes over time, comparing scenarios with and without CBDC.
    
    Args:
        model_no_cbdc: Model instance without CBDC
        model_with_cbdc: Model instance with CBDC
        window: Window size for moving average
    """
    # Get transaction data
    no_cbdc_data = model_no_cbdc.datacollector.get_model_vars_dataframe()
    with_cbdc_data = model_with_cbdc.datacollector.get_model_vars_dataframe()
    
    # Apply moving average for smoother lines
    no_cbdc_data_smooth = no_cbdc_data.rolling(window=window, min_periods=1).mean()
    with_cbdc_data_smooth = with_cbdc_data.rolling(window=window, min_periods=1).mean()
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot interbank volume
    ax1.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Interbank Volume'], 
             label='Without CBDC', color='steelblue', linewidth=2)
    ax1.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Interbank Volume'], 
             label='With CBDC', color='darkorange', linewidth=2)
    ax1.set_ylabel('Volume')
    ax1.set_title('Interbank Transaction Volume Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot bank transfer volumes
    ax2.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Same Bank Transfer Volume'], 
             label='Same Bank (No CBDC)', color='steelblue', linewidth=2)
    ax2.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Same Bank Transfer Volume'], 
             label='Same Bank (With CBDC)', color='darkorange', linewidth=2)
    ax2.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Interbank Transfer Volume'], 
             label='Interbank (No CBDC)', color='steelblue', linestyle='--', linewidth=2)
    ax2.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Interbank Transfer Volume'], 
             label='Interbank (With CBDC)', color='darkorange', linestyle='--', linewidth=2)
    ax2.set_ylabel('Volume')
    ax2.set_title('Bank Transfer Volumes Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot cash vs CBDC volume
    ax3.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Cash Volume'], 
             label='Cash (No CBDC)', color='steelblue', linewidth=2)
    ax3.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Cash Volume'], 
             label='Cash (With CBDC)', color='darkorange', linewidth=2)
    ax3.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['CBDC Volume'], 
             label='CBDC', color='forestgreen', linewidth=2)
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Volume')
    ax3.set_title('Cash vs CBDC Transaction Volume')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_centrality_metrics_over_time(model_no_cbdc, model_with_cbdc, window=5):
    """
    Plot centrality metrics over time, comparing scenarios with and without CBDC.
    
    Args:
        model_no_cbdc: Model instance without CBDC
        model_with_cbdc: Model instance with CBDC
        window: Window size for moving average
    """
    # Get centrality data
    no_cbdc_data = model_no_cbdc.datacollector.get_model_vars_dataframe()
    with_cbdc_data = model_with_cbdc.datacollector.get_model_vars_dataframe()
    
    # Apply moving average for smoother lines
    no_cbdc_data_smooth = no_cbdc_data.rolling(window=window, min_periods=1).mean()
    with_cbdc_data_smooth = with_cbdc_data.rolling(window=window, min_periods=1).mean()
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot degree centrality
    ax1.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Average Bank Degree Centrality'], 
             label='Without CBDC', color='steelblue', linewidth=2)
    ax1.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Average Bank Degree Centrality'], 
             label='With CBDC', color='darkorange', linewidth=2)
    ax1.set_ylabel('Average Degree Centrality')
    ax1.set_title('Bank Degree Centrality Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot betweenness centrality
    ax2.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Average Bank Betweenness Centrality'], 
             label='Without CBDC', color='steelblue', linewidth=2)
    ax2.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Average Bank Betweenness Centrality'], 
             label='With CBDC', color='darkorange', linewidth=2)
    ax2.set_ylabel('Average Betweenness Centrality')
    ax2.set_title('Bank Betweenness Centrality Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot eigenvector centrality
    ax3.plot(no_cbdc_data_smooth.index, no_cbdc_data_smooth['Average Bank Eigenvector Centrality'], 
             label='Without CBDC', color='steelblue', linewidth=2)
    ax3.plot(with_cbdc_data_smooth.index, with_cbdc_data_smooth['Average Bank Eigenvector Centrality'], 
             label='With CBDC', color='darkorange', linewidth=2)
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Average Eigenvector Centrality')
    ax3.set_title('Bank Eigenvector Centrality Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_network_structure_comparison(model_no_cbdc, model_with_cbdc):
    """
    Plot comparison of network structures with and without CBDC.
    
    Args:
        model_no_cbdc: Model instance without CBDC
        model_with_cbdc: Model instance with CBDC
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    plot_network(model_no_cbdc, title="Financial Network Without CBDC", ax=ax1)
    plot_network(model_with_cbdc, title="Financial Network With CBDC", ax=ax2)
    
    plt.tight_layout()
    return fig


def plot_bank_metrics_comparison(before_df, after_df):
    """
    Create a comparison plot of bank metrics before and after CBDC.
    
    Args:
        before_df: DataFrame with bank metrics before CBDC
        after_df: DataFrame with bank metrics after CBDC
    """
    # Prepare data
    comparison = pd.merge(before_df, after_df, on='bank_id', suffixes=('_before', '_after'))
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot degree centrality changes
    axs[0, 0].scatter(comparison['degree_centrality_before'], comparison['degree_centrality_after'], 
                     alpha=0.7, s=80, c='steelblue')
    max_val = max(comparison['degree_centrality_before'].max(), comparison['degree_centrality_after'].max()) * 1.1
    axs[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)  # Diagonal line
    axs[0, 0].set_xlabel('Degree Centrality Before CBDC')
    axs[0, 0].set_ylabel('Degree Centrality After CBDC')
    axs[0, 0].set_title('Change in Degree Centrality')
    axs[0, 0].grid(True, alpha=0.3)
    
    for i, bank_id in enumerate(comparison['bank_id']):
        axs[0, 0].annotate(f"{bank_id}", 
                          (comparison['degree_centrality_before'].iloc[i], 
                           comparison['degree_centrality_after'].iloc[i]),
                          fontsize=8)
    
    # Plot betweenness centrality changes
    axs[0, 1].scatter(comparison['betweenness_centrality_before'], comparison['betweenness_centrality_after'], 
                     alpha=0.7, s=80, c='darkorange')
    max_val = max(comparison['betweenness_centrality_before'].max(), 
                  comparison['betweenness_centrality_after'].max()) * 1.1
    axs[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)  # Diagonal line
    axs[0, 1].set_xlabel('Betweenness Centrality Before CBDC')
    axs[0, 1].set_ylabel('Betweenness Centrality After CBDC')
    axs[0, 1].set_title('Change in Betweenness Centrality')
    axs[0, 1].grid(True, alpha=0.3)
    
    for i, bank_id in enumerate(comparison['bank_id']):
        axs[0, 1].annotate(f"{bank_id}", 
                          (comparison['betweenness_centrality_before'].iloc[i], 
                           comparison['betweenness_centrality_after'].iloc[i]),
                          fontsize=8)
    
    # Plot deposit changes
    axs[1, 0].scatter(comparison['deposits_before'], comparison['deposits_after'], 
                     alpha=0.7, s=80, c='forestgreen')
    max_val = max(comparison['deposits_before'].max(), comparison['deposits_after'].max()) * 1.1
    axs[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)  # Diagonal line
    axs[1, 0].set_xlabel('Deposits Before CBDC')
    axs[1, 0].set_ylabel('Deposits After CBDC')
    axs[1, 0].set_title('Change in Bank Deposits')
    axs[1, 0].grid(True, alpha=0.3)
    
    for i, bank_id in enumerate(comparison['bank_id']):
        axs[1, 0].annotate(f"{bank_id}", 
                          (comparison['deposits_before'].iloc[i], 
                           comparison['deposits_after'].iloc[i]),
                          fontsize=8)
    
    # Plot customer changes
    axs[1, 1].scatter(comparison['num_customers_before'], comparison['num_customers_after'], 
                     alpha=0.7, s=80, c='purple')
    max_val = max(comparison['num_customers_before'].max(), comparison['num_customers_after'].max()) * 1.1
    axs[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)  # Diagonal line
    axs[1, 1].set_xlabel('Number of Customers Before CBDC')
    axs[1, 1].set_ylabel('Number of Customers After CBDC')
    axs[1, 1].set_title('Change in Bank Customer Base')
    axs[1, 1].grid(True, alpha=0.3)
    
    for i, bank_id in enumerate(comparison['bank_id']):
        axs[1, 1].annotate(f"{bank_id}", 
                          (comparison['num_customers_before'].iloc[i], 
                           comparison['num_customers_after'].iloc[i]),
                          fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_centrality_heatmap(comparison_df):
    """
    Create a heatmap showing percent changes in centrality metrics.
    
    Args:
        comparison_df: DataFrame with comparison metrics
    """
    # Extract centrality metrics
    metrics = ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality',
               'bank_degree_centrality', 'bank_betweenness_centrality', 'bank_eigenvector_centrality']
    
    # Create DataFrame with percent changes
    heatmap_data = comparison_df[['bank_id']].copy()
    for metric in metrics:
        heatmap_data[metric] = comparison_df[f'{metric}_pct_change']
    
    # Pivot the DataFrame for the heatmap
    pivot_data = heatmap_data.set_index('bank_id')
    
    # Define colors for negative and positive changes
    colors = ["steelblue", "white", "darkorange"]
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=100)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_data, cmap=cmap, center=0, annot=True, fmt=".1f", 
                    linewidths=0.5, cbar_kws={"label": "Percent Change (%)"})
    
    # Format axis labels
    ax.set_xticklabels([m.replace('_', ' ').title() for m in pivot_data.columns], rotation=45, ha='right')
    ax.set_title('Percent Change in Centrality Metrics After CBDC Introduction')
    
    plt.tight_layout()
    return plt.gcf()
