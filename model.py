"""
Defines the financial network model for CBDC impact simulation.
"""
from mesa import Model, time, datacollection, space
# Using direct imports for compatibility with different mesa versions
RandomActivation = time.RandomActivation
DataCollector = datacollection.DataCollector
NetworkGrid = space.NetworkGrid
import networkx as nx
import numpy as np
import random
import pandas as pd
from agents import CentralBank, CommercialBank, Individual
from network_analysis import calculate_network_metrics, get_model_network_metrics


class FinancialNetworkModel(Model):
    """
    Financial network model simulating the impact of CBDC on banking system.
    """

    def __init__(
        self,
        num_commercial_banks=20,
        num_individuals=1000,
        merchant_ratio=0.1,
        cbdc_active=False,
        random_seed=None
    ):
        super().__init__()
        self.num_commercial_banks = num_commercial_banks
        self.num_individuals = num_individuals
        self.merchant_ratio = merchant_ratio
        self.cbdc_active = cbdc_active
        self.schedule = RandomActivation(self)
        self.running = True
        self.transaction_log = []
        self.bank_interaction_log = []
        
        # Set random seed for reproducibility
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Create financial network
        self.G = nx.DiGraph()
        self.grid = NetworkGrid(self.G)
        
        # Setup data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Average Bank Degree Centrality": lambda m: get_model_network_metrics(m, "degree_centrality"),
                "Average Bank Betweenness Centrality": lambda m: get_model_network_metrics(m, "betweenness_centrality"),
                "Average Bank Eigenvector Centrality": lambda m: get_model_network_metrics(m, "eigenvector_centrality"),
                "Network Density": lambda m: nx.density(m.G),
                "Average Clustering Coefficient": lambda m: nx.average_clustering(m.G) if len(m.G.nodes()) > 2 else 0,
                "Interbank Volume": lambda m: sum(
                    transaction["amount"] for transaction in m.bank_interaction_log 
                    if transaction["step"] == m.schedule.steps
                ),
                "CBDC Volume": lambda m: sum(
                    transaction["amount"] for transaction in m.transaction_log 
                    if transaction["step"] == m.schedule.steps and transaction["method"] == "cbdc"
                ),
                "Cash Volume": lambda m: sum(
                    transaction["amount"] for transaction in m.transaction_log 
                    if transaction["step"] == m.schedule.steps and transaction["method"] == "cash"
                ),
                "Same Bank Transfer Volume": lambda m: sum(
                    transaction["amount"] for transaction in m.transaction_log 
                    if transaction["step"] == m.schedule.steps and transaction["method"] == "same_bank"
                ),
                "Interbank Transfer Volume": lambda m: sum(
                    transaction["amount"] for transaction in m.transaction_log 
                    if transaction["step"] == m.schedule.steps and transaction["method"] == "interbank"
                ),
            },
            agent_reporters={
                "Type": lambda a: a.type,
                "Degree Centrality": lambda a: a.degree_centrality if hasattr(a, "degree_centrality") else None,
                "Betweenness Centrality": lambda a: a.betweenness_centrality if hasattr(a, "betweenness_centrality") else None,
                "Eigenvector Centrality": lambda a: a.eigenvector_centrality if hasattr(a, "eigenvector_centrality") else None,
                "Capital": lambda a: a.capital if hasattr(a, "capital") else None,
                "Deposits": lambda a: a.deposits if hasattr(a, "deposits") else None,
                "Loans": lambda a: a.loans if hasattr(a, "loans") else None,
                "CBDC Holdings": lambda a: (
                    a.cbdc_holdings if hasattr(a, "cbdc_holdings") 
                    else a.cbdc_balance if hasattr(a, "cbdc_balance") 
                    else a.cbdc_issued if hasattr(a, "cbdc_issued") 
                    else None
                ),
            }
        )
        
        # Create agents
        self.create_agents()
        
        # Collect initial data
        self.datacollector.collect(self)
        
    def create_agents(self):
        """Create all agents in the simulation"""
        # Create central bank
        central_bank = CentralBank(self.next_id(), self, self.cbdc_active)
        self.schedule.add(central_bank)
        self.G.add_node(central_bank.unique_id)
        
        # Create commercial banks
        for _ in range(self.num_commercial_banks):
            bank = CommercialBank(self.next_id(), self, 
                                 initial_capital=np.random.uniform(8000, 12000))
            self.schedule.add(bank)
            self.G.add_node(bank.unique_id)
            
            # Connect to central bank
            self.G.add_edge(bank.unique_id, central_bank.unique_id, weight=1)
            self.G.add_edge(central_bank.unique_id, bank.unique_id, weight=1)
        
        # Create individual agents
        num_merchants = int(self.num_individuals * self.merchant_ratio)
        num_consumers = self.num_individuals - num_merchants
        
        # Create consumers
        for _ in range(num_consumers):
            individual = Individual(self.next_id(), self, is_merchant=False)
            self.schedule.add(individual)
            self.G.add_node(individual.unique_id)
            
            # Connect to primary bank if selected
            if individual.primary_bank_id is not None:
                self.G.add_edge(individual.unique_id, individual.primary_bank_id, weight=1)
                self.G.add_edge(individual.primary_bank_id, individual.unique_id, weight=1)
        
        # Create merchants
        for _ in range(num_merchants):
            merchant = Individual(self.next_id(), self, is_merchant=True)
            self.schedule.add(merchant)
            self.G.add_node(merchant.unique_id)
            
            # Connect to primary bank if selected
            if merchant.primary_bank_id is not None:
                self.G.add_edge(merchant.unique_id, merchant.primary_bank_id, weight=1)
                self.G.add_edge(merchant.primary_bank_id, merchant.unique_id, weight=1)
    
    def record_payment(self, sender_id, recipient_id, amount, method):
        """
        Record a payment between agents
        method can be: 'cash', 'cbdc', 'same_bank', 'interbank'
        """
        self.transaction_log.append({
            "step": self.schedule.steps,
            "sender": sender_id,
            "recipient": recipient_id,
            "amount": amount,
            "method": method
        })
        
        # Update network edge weight
        if self.G.has_edge(sender_id, recipient_id):
            # Increment weight of existing edge
            current_weight = self.G[sender_id][recipient_id]["weight"]
            self.G[sender_id][recipient_id]["weight"] = current_weight + amount
        else:
            # Add new edge
            self.G.add_edge(sender_id, recipient_id, weight=amount)
    
    def record_bank_interaction(self, sender_bank_id, recipient_bank_id, amount):
        """Record an interaction between banks"""
        self.bank_interaction_log.append({
            "step": self.schedule.steps,
            "sender_bank": sender_bank_id,
            "recipient_bank": recipient_bank_id,
            "amount": amount
        })
        
        # Update network edge weight
        if self.G.has_edge(sender_bank_id, recipient_bank_id):
            # Increment weight of existing edge
            current_weight = self.G[sender_bank_id][recipient_bank_id]["weight"]
            self.G[sender_bank_id][recipient_bank_id]["weight"] = current_weight + amount
        else:
            # Add new edge
            self.G.add_edge(sender_bank_id, recipient_bank_id, weight=amount)
    
    def step(self):
        """Advance the model by one step"""
        # Calculate network metrics for commercial banks
        calculate_network_metrics(self)
        
        # Execute agent actions
        self.schedule.step()
        
        # Collect data
        self.datacollector.collect(self)
    
    def get_transaction_history_df(self):
        """Get transaction history as a DataFrame"""
        return pd.DataFrame(self.transaction_log)
    
    def get_bank_interaction_history_df(self):
        """Get bank interaction history as a DataFrame"""
        return pd.DataFrame(self.bank_interaction_log)
