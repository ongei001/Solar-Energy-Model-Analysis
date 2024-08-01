import pandas as pd
import numpy as np  # Import NumPy library

from scipy import stats  # Importing the stats module for z-score calculation

# Use raw string literal or double backslashes to avoid syntax error
# Replace 'your_file_path.xlsx' with the actual path to your Excel file
file_path = r"C:\Users\carso\Documents\Personal project\Junior Data Analyst _ Data.xlsx"

# Read the Excel file into a DataFrame, skipping the first two rows
df = pd.read_excel(file_path, skiprows=2)

# Rename the columns in the DataFrame
df.columns = ['Hour', 'Date/Time Start', 'Solar Electricity Generation (kWh)', 'Electricity usage (kWh)']

# Convert 'Hour' column to numeric
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

# Drop rows with NaN values in the 'Hour' column
df.dropna(subset=['Hour'], inplace=True)

# Detect and remove outliers using z-score
z_scores = stats.zscore(df[['Solar Electricity Generation (kWh)', 'Electricity usage (kWh)']])
abs_z_scores = np.abs(z_scores)  # Use np.abs() for absolute values
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

# Display the first few rows of the DataFrame to check if the data is loaded correctly
print(df.head())

# Now, let's proceed with the visualization and analysis

import matplotlib.pyplot as plt


# Now, let's proceed with the visualization

# Calculate the average solar electricity generation and average electricity usage for each hour
average_solar_generation = df.groupby('Hour')['Solar Electricity Generation (kWh)'].mean()
average_electricity_usage = df.groupby('Hour')['Electricity usage (kWh)'].mean()

# Create a line plot for average solar electricity generation
plt.plot(average_solar_generation, label='Average Solar Electricity Generation')

# Create a line plot for average electricity usage
plt.plot(average_electricity_usage, label='Average Electricity Usage')

# Set labels and title
plt.xlabel('Hour')
plt.ylabel('kWh')
plt.title('Average Solar Electricity Generation and Electricity Usage by Hour')
plt.legend()

# Display the plot
plt.show()
import pandas as pd

# Use raw string literal or double backslashes to avoid syntax error
# Replace 'your_file_path.xlsx' with the actual path to your Excel file
file_path = r"C:\Users\carso\Documents\Personal project\Junior Data Analyst _ Data.xlsx"

# Read the Excel file into a DataFrame, skipping the first two rows
df = pd.read_excel(file_path, skiprows=2)

# Rename the columns in the DataFrame
df.columns = ['Hour', 'Date/Time Start', 'Solar Electricity Generation (kWh)', 'Electricity usage (kWh)']

# Convert 'Hour' column to numeric
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

# Drop rows with NaN values in the 'Hour' column
df.dropna(subset=['Hour'], inplace=True)

# Calculate the amount of electricity that needed to be bought from the electricity provider for each hour in 2020
df['Net Electricity Consumption (kWh)'] = df['Electricity usage (kWh)'] - df['Solar Electricity Generation (kWh)']
df['Net Electricity Consumption (kWh)'] = df['Net Electricity Consumption (kWh)'].apply(lambda x: max(x, 0))
total_electricity_bought = df.groupby('Hour')['Net Electricity Consumption (kWh)'].sum()

# Display the total electricity bought for each hour
print(total_electricity_bought)

# Step 2(iii): Calculate the excess solar electricity generated over electricity used for each hour in 2020

# Calculate excess solar electricity generated over electricity used for each hour
df['Excess Solar Electricity (kWh)'] = df['Solar Electricity Generation (kWh)'] - df['Electricity usage (kWh)']

# Set negative values to zero
df['Excess Solar Electricity (kWh)'] = df['Excess Solar Electricity (kWh)'].apply(lambda x: max(x, 0))

# Summarize excess solar electricity generated for each hour over the entire year
total_excess_solar_electricity = df.groupby('Hour')['Excess Solar Electricity (kWh)'].sum()

# Display the total excess solar electricity generated for each hour
print(total_excess_solar_electricity)
# Step 2(iv): Model the cumulative battery charge level for each hour over 2020, assuming a battery had already been installed

# Define battery capacity (in kWh) and initialize cumulative charge level
battery_capacity = 12.5  # kWh
cumulative_charge_level = 0

# Initialize a list to store the cumulative charge level for each hour
cumulative_charge_levels = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Calculate net electricity consumption for the current hour
    net_electricity_consumption = row['Electricity usage (kWh)'] - row['Solar Electricity Generation (kWh)']
    
    # Calculate the change in charge level based on net electricity consumption and battery capacity constraints
    charge_change = min(net_electricity_consumption, battery_capacity - cumulative_charge_level)
    
    # Update the cumulative charge level
    cumulative_charge_level += charge_change
    
    # Append the cumulative charge level to the list
    cumulative_charge_levels.append(cumulative_charge_level)

# Add the cumulative charge level to the DataFrame
df['Cumulative Battery Charge Level (kWh)'] = cumulative_charge_levels

# Display the DataFrame with the cumulative charge level
print(df[['Hour', 'Cumulative Battery Charge Level (kWh)']])
# Step 2(v): Calculate the amount of electricity for each hour in 2020 that would have been bought from the electricity provider, assuming a battery had already been installed

# Calculate net electricity consumption after considering the battery's discharge
df['Net Electricity Consumption After Battery (kWh)'] = df['Net Electricity Consumption (kWh)'] - df['Cumulative Battery Charge Level (kWh)']

# Replace negative values (indicating excess charge in the battery) with zero
df['Net Electricity Consumption After Battery (kWh)'] = df['Net Electricity Consumption After Battery (kWh)'].apply(lambda x: max(x, 0))

# Calculate the amount of electricity bought from the electricity provider for each hour
df['Electricity Bought from Provider (kWh)'] = df['Net Electricity Consumption After Battery (kWh)']

# Display the DataFrame with the calculated values
print(df[['Hour', 'Electricity Bought from Provider (kWh)']])
# Step 2(vi): Calculate the savings over 2020 (in dollars) from installing a battery compared to using the existing solar panels alone

# Calculate the total electricity costs without the battery
total_costs_without_battery = df['Electricity Bought from Provider (kWh)'].sum() * 0.17

# Calculate the total electricity costs with the battery
total_costs_with_battery = df['Net Electricity Consumption (kWh)'].sum() * 0.17

# Calculate the savings from installing a battery
savings = total_costs_without_battery - total_costs_with_battery

# Display the calculated savings
print("Savings from installing a battery compared to using solar panels alone: ${:.2f}".format(savings))
# Step 2(vii): Tabulate the data appropriately and produce a chart


# Convert the 'Date/hour start' column to datetime format
df['Date/Time Start'] = pd.to_datetime(df['Date/Time Start'])

# Extract month and year from the 'Date/hour start' column
df['Month'] = df['Date/Time Start'].dt.month
df['Year'] = df['Date/Time Start'].dt.year

# Group the data by month and year and calculate the sum of each column
monthly_data = df.groupby(['Year', 'Month']).agg({
    'Solar Electricity Generation (kWh)': 'sum',
    'Electricity usage (kWh)': 'sum',
    'Electricity Bought from Provider (kWh)': 'sum',
    'Net Electricity Consumption (kWh)': 'sum'
}).reset_index()

# Rename columns for clarity
monthly_data.columns = ['Year', 'Month', 'Monthly Solar Generation (kWh)', 'Monthly Electricity Usage (kWh)',
                        'Monthly Electricity Purchased (No Battery) (kWh)', 'Monthly Electricity Purchased (With Battery) (kWh)']

# Plotting the data
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(12, 6))

# Plot monthly solar generation
plt.plot(monthly_data['Month'], monthly_data['Monthly Solar Generation (kWh)'], label='Solar Generation')

# Plot monthly electricity usage
plt.plot(monthly_data['Month'], monthly_data['Monthly Electricity Usage (kWh)'], label='Electricity Usage')

# Plot monthly electricity purchased without battery
plt.plot(monthly_data['Month'], monthly_data['Monthly Electricity Purchased (No Battery) (kWh)'],
         label='Electricity Purchased (No Battery)')

# Plot monthly electricity purchased with battery
plt.plot(monthly_data['Month'], monthly_data['Monthly Electricity Purchased (With Battery) (kWh)'],
         label='Electricity Purchased (With Battery)')

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Electricity (kWh)')
plt.title('Monthly Electricity Data')
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)

# Show plot
plt.show()
import numpy_financial as npf

# Define function to calculate future annual savings
def calculate_future_savings(initial_year, num_years, electricity_price_increase, battery_savings):
    # Initialize lists to store savings and discount factors
    savings = []
    discount_factors = []

    # Iterate over each year
    for year in range(initial_year, initial_year + num_years):
        # Calculate savings for the current year
        savings.append(battery_savings * electricity_price_increase ** (year - initial_year))
        
        # Calculate discount factor for the current year
        discount_factors.append(1 / (1 + discount_rate) ** (year - initial_year))

    return savings, discount_factors

# Define function to calculate NPV
def calculate_npv(savings, discount_factors):
    # Calculate NPV using the formula: NPV = sum(savings / discount_factors)
    npv = sum(savings[i] * discount_factors[i] for i in range(len(savings)))
    return npv

# Define function to calculate IRR
def calculate_irr(savings, initial_investment):
    # Calculate IRR using numpy_financial's financial functions
    irr = npf.irr([-initial_investment] + savings)
    return irr

# Set parameters
initial_year = 2022
num_years = 20
discount_rate = 0.06  # Discount rate for NPV calculation
electricity_price_increase_government = 0.04
electricity_price_increase_naomi = 0.04 + 0.0025  # 0.25% increase each year
battery_savings = 10000  # Example value representing the savings achieved by installing the battery
battery_cost = 20000     # Example value representing the cost of purchasing and installing the battery

# Calculate future annual savings for both scenarios
savings_government, discount_factors = calculate_future_savings(initial_year, num_years,
                                                               electricity_price_increase_government,battery_savings)
savings_naomi, _ = calculate_future_savings(initial_year, num_years,
                                            electricity_price_increase_naomi,
                                            battery_savings)

# Calculate NPV for both scenarios
npv_government = calculate_npv(savings_government, discount_factors)
npv_naomi = calculate_npv(savings_naomi, discount_factors)

# Calculate IRR for both scenarios
irr_government = calculate_irr(savings_government, battery_cost)
irr_naomi = calculate_irr(savings_naomi, battery_cost)

# Print results
print("NPV for government scenario:", npv_government)
print("NPV for Naomi's scenario:", npv_naomi)
print("IRR for government scenario:", irr_government)
print("IRR for Naomi's scenario:", irr_naomi)
