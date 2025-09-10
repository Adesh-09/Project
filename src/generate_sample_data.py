import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_churn_data(n_samples=10000, seed=42):
    """
    Generate synthetic customer churn data for testing and demonstration
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Define possible values for categorical features
    genders = ['Male', 'Female']
    yes_no = ['Yes', 'No']
    internet_services = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    
    data = []
    
    for i in range(n_samples):
        customer_id = f"CUST_{i+1:06d}"
        
        # Basic demographics
        gender = random.choice(genders)
        senior_citizen = np.random.choice([0, 1], p=[0.84, 0.16])  # 16% senior citizens
        partner = random.choice(yes_no)
        dependents = random.choice(yes_no)
        
        # Tenure (months with company)
        tenure = np.random.randint(1, 73)  # 1 to 72 months
        
        # Services
        phone_service = random.choice(yes_no)
        multiple_lines = random.choice(yes_no + ['No phone service'])
        internet_service = random.choice(internet_services)
        
        # Internet-dependent services
        if internet_service == 'No':
            online_security = 'No internet service'
            online_backup = 'No internet service'
            device_protection = 'No internet service'
            tech_support = 'No internet service'
            streaming_tv = 'No internet service'
            streaming_movies = 'No internet service'
        else:
            online_security = random.choice(yes_no + ['No internet service'])
            online_backup = random.choice(yes_no + ['No internet service'])
            device_protection = random.choice(yes_no + ['No internet service'])
            tech_support = random.choice(yes_no + ['No internet service'])
            streaming_tv = random.choice(yes_no + ['No internet service'])
            streaming_movies = random.choice(yes_no + ['No internet service'])
        
        # Contract and billing
        contract = random.choice(contracts)
        paperless_billing = random.choice(yes_no)
        payment_method = random.choice(payment_methods)
        
        # Charges
        base_charge = 20
        if phone_service == 'Yes':
            base_charge += 10
        if multiple_lines == 'Yes':
            base_charge += 10
        if internet_service == 'DSL':
            base_charge += 20
        elif internet_service == 'Fiber optic':
            base_charge += 40
        
        # Add charges for additional services
        services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
        additional_services = sum(1 for service in services if service == 'Yes')
        base_charge += additional_services * 5
        
        # Add some randomness to monthly charges
        monthly_charges = base_charge + np.random.normal(0, 5)
        monthly_charges = max(monthly_charges, 18.25)  # Minimum charge
        monthly_charges = round(monthly_charges, 2)
        
        # Total charges based on tenure and monthly charges
        total_charges = monthly_charges * tenure + np.random.normal(0, 50)
        total_charges = max(total_charges, monthly_charges)  # At least one month
        total_charges = round(total_charges, 2)
        
        # Churn probability based on various factors
        churn_prob = 0.1  # Base probability
        
        # Factors that increase churn probability
        if contract == 'Month-to-month':
            churn_prob += 0.3
        if senior_citizen == 1:
            churn_prob += 0.1
        if partner == 'No':
            churn_prob += 0.1
        if dependents == 'No':
            churn_prob += 0.05
        if tenure < 12:
            churn_prob += 0.2
        if payment_method == 'Electronic check':
            churn_prob += 0.15
        if monthly_charges > 80:
            churn_prob += 0.1
        if paperless_billing == 'Yes':
            churn_prob += 0.05
        
        # Factors that decrease churn probability
        if contract == 'Two year':
            churn_prob -= 0.2
        if tenure > 36:
            churn_prob -= 0.15
        if additional_services > 3:
            churn_prob -= 0.1
        
        # Ensure probability is between 0 and 1
        churn_prob = max(0, min(1, churn_prob))
        
        # Determine churn
        churn = 'Yes' if np.random.random() < churn_prob else 'No'
        
        # Create customer record
        customer = {
            'CustomerID': customer_id,
            'Gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Churn': churn
        }
        
        data.append(customer)
    
    return pd.DataFrame(data)

def generate_time_series_data(base_df, months=12):
    """
    Generate time series data for model retraining simulation
    """
    time_series_data = []
    
    for month in range(months):
        # Create a copy of base data with some drift
        month_data = base_df.copy()
        
        # Simulate data drift over time
        drift_factor = month * 0.02  # 2% drift per month
        
        # Adjust monthly charges (inflation)
        month_data['MonthlyCharges'] = month_data['MonthlyCharges'] * (1 + drift_factor)
        
        # Adjust churn rate (market changes)
        churn_adjustment = np.random.normal(0, 0.01, len(month_data))
        for i, (idx, row) in enumerate(month_data.iterrows()):
            if row['Churn'] == 'No' and churn_adjustment[i] > 0.05:
                month_data.at[idx, 'Churn'] = 'Yes'
            elif row['Churn'] == 'Yes' and churn_adjustment[i] < -0.05:
                month_data.at[idx, 'Churn'] = 'No'
        
        # Add timestamp
        month_data['DataMonth'] = datetime.now() - timedelta(days=30*(months-month))
        
        time_series_data.append(month_data)
    
    return pd.concat(time_series_data, ignore_index=True)

if __name__ == "__main__":
    print("Generating sample customer churn data...")
    
    # Generate main dataset
    df = generate_customer_churn_data(n_samples=10000)
    
    # Save to CSV
    df.to_csv('/home/ubuntu/projects/customer_churn_prediction/data/telecom_churn.csv', index=False)
    print(f"Generated {len(df)} customer records")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    
    # Generate smaller dataset for testing
    test_df = generate_customer_churn_data(n_samples=1000, seed=123)
    test_df.to_csv('/home/ubuntu/projects/customer_churn_prediction/data/test_churn.csv', index=False)
    
    # Generate time series data for retraining simulation
    ts_df = generate_time_series_data(df.sample(1000), months=6)
    ts_df.to_csv('/home/ubuntu/projects/customer_churn_prediction/data/time_series_churn.csv', index=False)
    
    # Create reference data for drift detection
    reference_df = df.sample(5000, random_state=42)
    reference_df.to_csv('/home/ubuntu/projects/customer_churn_prediction/data/reference_churn.csv', index=False)
    
    print("Sample data generation completed!")
    print("\nDataset statistics:")
    print(f"Total customers: {len(df)}")
    print(f"Churned customers: {(df['Churn'] == 'Yes').sum()}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print(f"Average tenure: {df['Tenure'].mean():.1f} months")
    print(f"Average monthly charges: ${df['MonthlyCharges'].mean():.2f}")
    print(f"Senior citizens: {(df['SeniorCitizen'] == 1).mean():.1%}")
    
    # Display sample records
    print("\nSample records:")
    print(df.head())

