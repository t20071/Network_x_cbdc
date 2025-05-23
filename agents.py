"""
Defines the agent types for the CBDC impact financial network simulation.
Three agent types: CentralBank, CommercialBank, and Individual.
"""
from mesa import Agent
import numpy as np
import random

# -------------------- Central Bank Agent --------------------
class CentralBank(Agent):
    """
    Central Bank agent that issues physical cash and CBDC.
    Controls monetary policy and interacts with commercial banks.
    """
    def __init__(self, unique_id, model, cbdc_active=False):
        super().__init__(unique_id, model)
        self.type = "central_bank"
        self.reserves = 1000000  # Central bank initial reserves
        self.cbdc_active = cbdc_active  # Whether CBDC is active
        self.cbdc_issued = 0  # Amount of CBDC issued
        self.physical_cash_issued = 0  # Amount of physical cash issued
        self.interest_rate = 0.02  # Base interest rate

    def issue_physical_cash(self, amount):
        """Issue physical cash to the economy"""
        self.physical_cash_issued += amount
        self.reserves -= amount
        return amount

    def issue_cbdc(self, amount):
        """Issue CBDC to the economy"""
        if not self.cbdc_active:
            return 0
        
        self.cbdc_issued += amount
        self.reserves -= amount
        return amount

    def set_interest_rate(self, rate):
        """Set the base interest rate"""
        self.interest_rate = rate

    def step(self):
        """Central bank policy decisions on each step"""
        # Adjust interest rate slightly based on economic conditions
        if random.random() < 0.1:  # 10% chance of rate change
            rate_change = np.random.normal(0, 0.002)  # Small random adjustment
            self.interest_rate = max(0.001, self.interest_rate + rate_change)
            self.model.logger.info(f"Central bank adjusted interest rate to {self.interest_rate:.4f}")


class CommercialBank(Agent):
    """
    Commercial Bank agent that handles deposits, loans, and payments.
    Intermediates transactions between individuals and other banks.
    """
    def __init__(self, unique_id, model, initial_capital=10000):
        super().__init__(unique_id, model)
        self.type = "commercial_bank"
        self.capital = initial_capital  # Bank's own capital
        self.deposits = 0  # Total customer deposits
        self.loans = 0  # Total loans issued
        self.reserves_at_central_bank = 0  # Reserves held at central bank
        self.interbank_loans = {}  # Loans to other banks {bank_id: amount}
        self.interbank_borrowings = {}  # Borrowings from other banks {bank_id: amount}
        self.liquidity_ratio = 0.1  # Required liquidity ratio
        self.default_risk = np.random.uniform(0.01, 0.05)  # Probability of loan default
        self.cbdc_holdings = 0  # CBDC holdings if CBDC is active
        self.customers = []  # List of customer IDs
        
        # Network metrics (to be calculated)
        self.degree_centrality = 0
        self.betweenness_centrality = 0
        self.eigenvector_centrality = 0

    def accept_deposit(self, amount, customer_id):
        """Accept deposit from a customer"""
        self.deposits += amount
        if customer_id not in self.customers:
            self.customers.append(customer_id)
        return True

    def withdraw(self, amount, customer_id):
        """Process withdrawal request"""
        if customer_id not in self.customers:
            return False
        
        # Check if bank has enough liquidity
        available_funds = self.capital + self.deposits - self.loans
        if available_funds >= amount:
            self.deposits -= amount
            return True
        else:
            # Try to borrow from other banks or central bank
            if self.borrow_from_interbank_market(amount - available_funds):
                self.deposits -= amount
                return True
            return False

    def make_loan(self, amount, customer_id):
        """Issue a loan to a customer"""
        # Check if bank has enough capital to make the loan
        # (simplified capital adequacy)
        if (self.loans + amount) <= (self.capital * 10):  # Leverage limit
            self.loans += amount
            if customer_id not in self.customers:
                self.customers.append(customer_id)
            return True
        return False

    def transfer_to_other_bank(self, amount, receiving_bank_id, customer_id):
        """Transfer funds to another bank"""
        if customer_id not in self.customers:
            return False
            
        # Find the receiving bank
        receiving_bank = None
        for bank in self.model.schedule.agents:
            if bank.unique_id == receiving_bank_id and bank.type == "commercial_bank":
                receiving_bank = bank
                break
                
        if receiving_bank is None:
            return False
            
        # Process the transfer
        if self.withdraw(amount, customer_id):
            receiving_bank.deposits += amount
            
            # Record the interaction for network analysis
            self.model.record_bank_interaction(self.unique_id, receiving_bank_id, amount)
            return True
        return False

    def borrow_from_interbank_market(self, amount):
        """Attempt to borrow from other banks"""
        # Simplified interbank lending logic
        available_lenders = []
        for bank in self.model.schedule.agents:
            if (bank.type == "commercial_bank" and 
                bank.unique_id != self.unique_id and
                (bank.capital + bank.deposits - bank.loans) > amount * 1.2):  # Lender has excess liquidity
                available_lenders.append(bank)
                
        if not available_lenders:
            return False
            
        # Choose a random lender
        lender = random.choice(available_lenders)
        
        # Interest rate based on central bank rate plus risk premium
        central_bank = next(a for a in self.model.schedule.agents if a.type == "central_bank")
        interest_rate = central_bank.interest_rate + self.default_risk
        
        # Record the loan
        lender.interbank_loans[self.unique_id] = lender.interbank_loans.get(self.unique_id, 0) + amount
        self.interbank_borrowings[lender.unique_id] = self.interbank_borrowings.get(lender.unique_id, 0) + amount
        
        # Update lender's and borrower's accounts
        lender.loans += amount
        self.capital += amount
        
        # Record the interaction for network analysis
        self.model.record_bank_interaction(lender.unique_id, self.unique_id, amount)
        
        return True

    def step(self):
        """Bank operations on each step"""
        # Process interest on deposits and loans
        central_bank = next(a for a in self.model.schedule.agents if a.type == "central_bank")
        
        # Earn interest on loans
        loan_interest = self.loans * (central_bank.interest_rate + 0.03)  # 3% spread
        
        # Pay interest on deposits
        deposit_interest = self.deposits * max(0, central_bank.interest_rate - 0.01)  # 1% below central bank rate
        
        # Calculate net interest income
        net_interest = loan_interest - deposit_interest
        
        # Apply loan defaults
        loan_defaults = self.loans * random.uniform(0, self.default_risk)
        
        # Update capital
        self.capital += net_interest - loan_defaults
        
        # Manage reserves at central bank (simplified)
        target_reserves = self.deposits * self.liquidity_ratio
        if self.reserves_at_central_bank < target_reserves:
            # Increase reserves if needed
            deficit = target_reserves - self.reserves_at_central_bank
            if self.capital >= deficit:
                self.capital -= deficit
                self.reserves_at_central_bank += deficit
        
        # Check if CBDC is active and adjust strategy
        if central_bank.cbdc_active and random.random() < 0.3:  # 30% chance of CBDC strategy adjustment
            # Acquire some CBDC as part of liquidity management
            cbdc_adjustment = random.uniform(-0.1, 0.2) * self.cbdc_holdings  # Adjust holdings up or down
            if cbdc_adjustment > 0 and self.capital > cbdc_adjustment:
                self.capital -= cbdc_adjustment
                self.cbdc_holdings += cbdc_adjustment
            elif cbdc_adjustment < 0:
                self.capital -= cbdc_adjustment  # Adding negative value
                self.cbdc_holdings += cbdc_adjustment


class Individual(Agent):
    """
    Individual agent representing consumers and merchants in the economy.
    Performs transactions, holds deposits, and may use CBDC.
    """
    def __init__(self, unique_id, model, is_merchant=False):
        super().__init__(unique_id, model)
        self.type = "individual"
        self.is_merchant = is_merchant
        self.cash = np.random.lognormal(4, 1) if not is_merchant else np.random.lognormal(6, 1.5)
        self.bank_deposits = {}  # {bank_id: amount}
        self.loans = {}  # {bank_id: amount}
        self.cbdc_balance = 0  # CBDC holdings if CBDC is active
        self.income = np.random.lognormal(4, 1) / 26  # Bi-weekly income
        self.spending_rate = np.random.uniform(0.3, 0.7)  # Portion of income spent
        self.saving_rate = np.random.uniform(0.1, 0.4)  # Portion of income saved
        self.cbdc_preference = np.random.uniform(0, 1)  # Preference for CBDC vs traditional banking
        
        # Choose a primary bank
        self.primary_bank_id = None
        self.select_primary_bank()

    def select_primary_bank(self):
        """Select a primary bank from available commercial banks"""
        commercial_banks = [a for a in self.model.schedule.agents if a.type == "commercial_bank"]
        if commercial_banks:
            bank = random.choice(commercial_banks)
            self.primary_bank_id = bank.unique_id
            
            # Make initial deposit
            initial_deposit = self.cash * 0.6  # Deposit 60% of cash
            self.cash -= initial_deposit
            self.bank_deposits[bank.unique_id] = initial_deposit
            bank.accept_deposit(initial_deposit, self.unique_id)

    def make_payment(self, amount, recipient_id):
        """Make a payment to another individual"""
        # Find the recipient
        recipient = None
        for agent in self.model.schedule.agents:
            if agent.unique_id == recipient_id and agent.type == "individual":
                recipient = agent
                break
                
        if recipient is None:
            return False
        
        central_bank = next(a for a in self.model.schedule.agents if a.type == "central_bank")
        
        # Payment method selection
        # 1. CBDC if available and preferred
        if central_bank.cbdc_active and self.cbdc_balance >= amount and self.cbdc_preference > 0.5:
            self.cbdc_balance -= amount
            recipient.cbdc_balance += amount
            self.model.record_payment(self.unique_id, recipient_id, amount, "cbdc")
            return True
            
        # 2. Cash payment
        elif self.cash >= amount:
            self.cash -= amount
            recipient.cash += amount
            self.model.record_payment(self.unique_id, recipient_id, amount, "cash")
            return True
            
        # 3. Bank transfer (same bank)
        elif recipient.primary_bank_id == self.primary_bank_id:
            if self.bank_deposits.get(self.primary_bank_id, 0) >= amount:
                self.bank_deposits[self.primary_bank_id] -= amount
                recipient.bank_deposits[self.primary_bank_id] = recipient.bank_deposits.get(self.primary_bank_id, 0) + amount
                self.model.record_payment(self.unique_id, recipient_id, amount, "same_bank")
                return True
                
        # 4. Bank transfer (different banks)
        else:
            sender_bank_id = self.primary_bank_id
            recipient_bank_id = recipient.primary_bank_id
            
            if (sender_bank_id is not None and 
                recipient_bank_id is not None and 
                self.bank_deposits.get(sender_bank_id, 0) >= amount):
                
                # Find sender's bank
                sender_bank = None
                for bank in self.model.schedule.agents:
                    if bank.unique_id == sender_bank_id and bank.type == "commercial_bank":
                        sender_bank = bank
                        break
                
                # Find recipient's bank
                recipient_bank = None
                for bank in self.model.schedule.agents:
                    if bank.unique_id == recipient_bank_id and bank.type == "commercial_bank":
                        recipient_bank = bank
                        break
                        
                if sender_bank and recipient_bank:
                    # Process the interbank transfer
                    if sender_bank.transfer_to_other_bank(amount, recipient_bank_id, self.unique_id):
                        self.bank_deposits[sender_bank_id] -= amount
                        recipient.bank_deposits[recipient_bank_id] = recipient.bank_deposits.get(recipient_bank_id, 0) + amount
                        self.model.record_payment(self.unique_id, recipient_id, amount, "interbank")
                        return True
        
        return False

    def step(self):
        """Individual activities on each step"""
        # Receive income
        self.cash += self.income
        
        # Deposit some cash to bank
        if self.primary_bank_id is not None and self.cash > 0:
            deposit_amount = self.cash * self.saving_rate
            if deposit_amount > 0:
                self.cash -= deposit_amount
                self.bank_deposits[self.primary_bank_id] = self.bank_deposits.get(self.primary_bank_id, 0) + deposit_amount
                
                # Find primary bank
                primary_bank = None
                for bank in self.model.schedule.agents:
                    if bank.unique_id == self.primary_bank_id and bank.type == "commercial_bank":
                        primary_bank = bank
                        break
                        
                if primary_bank:
                    primary_bank.accept_deposit(deposit_amount, self.unique_id)
        
        # Spend money (payments to other individuals)
        if not self.is_merchant:  # Merchants receive payments, don't initiate them
            spending_amount = self.income * self.spending_rate
            if spending_amount > 0:
                # Find potential merchant recipients
                merchants = [a for a in self.model.schedule.agents if a.type == "individual" and a.is_merchant]
                if merchants:
                    # Make 1-3 payments to random merchants
                    num_payments = random.randint(1, 3)
                    payment_recipients = random.sample(merchants, min(num_payments, len(merchants)))
                    payment_size = spending_amount / len(payment_recipients)
                    
                    for recipient in payment_recipients:
                        self.make_payment(payment_size, recipient.unique_id)
        
        # Handle CBDC adjustments if active
        central_bank = next(a for a in self.model.schedule.agents if a.type == "central_bank")
        if central_bank.cbdc_active:
            # Potentially convert some cash or bank deposits to CBDC based on preference
            if self.cbdc_preference > 0.5:  # Higher preference for CBDC
                if random.random() < 0.3:  # 30% chance of adjustment
                    if self.cash > 0:
                        # Convert some cash to CBDC
                        convert_amount = self.cash * random.uniform(0.1, 0.3)
                        self.cash -= convert_amount
                        self.cbdc_balance += convert_amount
                    elif self.primary_bank_id is not None and self.bank_deposits.get(self.primary_bank_id, 0) > 0:
                        # Withdraw from bank to CBDC
                        convert_amount = self.bank_deposits[self.primary_bank_id] * random.uniform(0.05, 0.15)
                        
                        # Find primary bank
                        primary_bank = None
                        for bank in self.model.schedule.agents:
                            if bank.unique_id == self.primary_bank_id and bank.type == "commercial_bank":
                                primary_bank = bank
                                break
                                
                        if primary_bank and primary_bank.withdraw(convert_amount, self.unique_id):
                            self.bank_deposits[self.primary_bank_id] -= convert_amount
                            self.cbdc_balance += convert_amount
