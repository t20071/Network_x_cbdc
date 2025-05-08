"""
Network analysis functions for the financial network simulation.
"""
import networkx as nx
import pandas as pd


def calculate_network_metrics(model):
    """
    Calculate network centrality metrics for commercial banks.
    Updates the centrality attributes for each commercial bank agent.
    
    Args:
        model: The financial network model instance
    """
    # Get the network
    G = model.G
    
    # Create a subgraph of just banks for some metrics
    bank_agents = [agent for agent in model.schedule.agents 
                 if agent.type == "commercial_bank" or agent.type == "central_bank"]
    bank_ids = [agent.unique_id for agent in bank_agents]
    bank_subgraph = G.subgraph(bank_ids)
    
    # Ensure network has at least one node
    if len(G.nodes()) == 0:
        return
    
    # Calculate various centrality metrics
    try:
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness centrality (may be computationally expensive for large networks)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
        
        # Eigenvector centrality 
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
        
        # Bank-specific metrics using the bank subgraph
        if len(bank_subgraph.nodes()) > 1:
            bank_degree_centrality = nx.degree_centrality(bank_subgraph)
            bank_betweenness_centrality = nx.betweenness_centrality(bank_subgraph, weight='weight')
            bank_eigenvector_centrality = nx.eigenvector_centrality_numpy(bank_subgraph, weight='weight')
        else:
            bank_degree_centrality = {node: 0 for node in bank_subgraph.nodes()}
            bank_betweenness_centrality = {node: 0 for node in bank_subgraph.nodes()}
            bank_eigenvector_centrality = {node: 0 for node in bank_subgraph.nodes()}
        
        # Update commercial bank agents with their metrics
        for agent in model.schedule.agents:
            if agent.type == "commercial_bank":
                # Full network metrics
                agent.degree_centrality = degree_centrality.get(agent.unique_id, 0)
                agent.betweenness_centrality = betweenness_centrality.get(agent.unique_id, 0)
                agent.eigenvector_centrality = eigenvector_centrality.get(agent.unique_id, 0)
                
                # Bank subgraph metrics
                agent.bank_degree_centrality = bank_degree_centrality.get(agent.unique_id, 0)
                agent.bank_betweenness_centrality = bank_betweenness_centrality.get(agent.unique_id, 0)
                agent.bank_eigenvector_centrality = bank_eigenvector_centrality.get(agent.unique_id, 0)
    
    except Exception as e:
        print(f"Error calculating network metrics: {e}")
        # Set default values in case of calculation error
        for agent in model.schedule.agents:
            if agent.type == "commercial_bank":
                agent.degree_centrality = 0
                agent.betweenness_centrality = 0
                agent.eigenvector_centrality = 0
                agent.bank_degree_centrality = 0
                agent.bank_betweenness_centrality = 0
                agent.bank_eigenvector_centrality = 0


def get_model_network_metrics(model, metric_name):
    """
    Get average centrality metrics for commercial banks.
    
    Args:
        model: The financial network model instance
        metric_name: Name of the metric to get (e.g., 'degree_centrality')
    
    Returns:
        float: Average value of the specified metric across commercial banks
    """
    commercial_banks = [agent for agent in model.schedule.agents if agent.type == "commercial_bank"]
    
    if not commercial_banks:
        return 0
    
    # Sum the specified metric across all commercial banks
    total_metric = 0
    for bank in commercial_banks:
        total_metric += getattr(bank, metric_name, 0)
    
    # Return the average
    return total_metric / len(commercial_banks)


def get_network_metrics_df(model):
    """
    Get a DataFrame containing network metrics for all commercial banks.
    
    Args:
        model: The financial network model instance
    
    Returns:
        pandas.DataFrame: DataFrame with network metrics for each bank
    """
    commercial_banks = [agent for agent in model.schedule.agents if agent.type == "commercial_bank"]
    
    data = []
    for bank in commercial_banks:
        bank_data = {
            'bank_id': bank.unique_id,
            'degree_centrality': bank.degree_centrality,
            'betweenness_centrality': bank.betweenness_centrality,
            'eigenvector_centrality': bank.eigenvector_centrality,
            'bank_degree_centrality': getattr(bank, 'bank_degree_centrality', 0),
            'bank_betweenness_centrality': getattr(bank, 'bank_betweenness_centrality', 0),
            'bank_eigenvector_centrality': getattr(bank, 'bank_eigenvector_centrality', 0),
            'capital': bank.capital,
            'deposits': bank.deposits,
            'loans': bank.loans,
            'cbdc_holdings': bank.cbdc_holdings,
            'num_customers': len(bank.customers)
        }
        data.append(bank_data)
    
    return pd.DataFrame(data)


def compare_network_metrics(before_df, after_df):
    """
    Compare network metrics before and after CBDC introduction.
    
    Args:
        before_df: DataFrame with metrics before CBDC
        after_df: DataFrame with metrics after CBDC
    
    Returns:
        pandas.DataFrame: DataFrame with comparison of metrics
    """
    # Merge DataFrames on bank_id
    comparison = pd.merge(before_df, after_df, on='bank_id', suffixes=('_before', '_after'))
    
    # Calculate differences and percent changes
    metrics = ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality',
              'bank_degree_centrality', 'bank_betweenness_centrality', 'bank_eigenvector_centrality',
              'capital', 'deposits', 'loans', 'num_customers']
    
    for metric in metrics:
        comparison[f'{metric}_diff'] = comparison[f'{metric}_after'] - comparison[f'{metric}_before']
        comparison[f'{metric}_pct_change'] = (
            (comparison[f'{metric}_after'] - comparison[f'{metric}_before']) / 
            comparison[f'{metric}_before'] * 100
        ).fillna(0)
    
    return comparison


def get_community_structure(model):
    """
    Identify communities within the financial network.
    
    Args:
        model: The financial network model instance
    
    Returns:
        dict: Community membership for each node
    """
    # Use Louvain method for community detection
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(model.G.to_undirected())
        return partition
    except ImportError:
        # Fallback to NetworkX's built-in methods
        communities = list(nx.algorithms.community.greedy_modularity_communities(model.G.to_undirected()))
        partition = {}
        for i, community in enumerate(communities):
            for node in community:
                partition[node] = i
        return partition
