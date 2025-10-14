import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import streamlit as st
import io

# --- Data cleaning and parsing utility functions ---

def starting_time(df):

    first_processing = df.loc[0, 'Date from']
    first_wash = df.loc[0,'First Wash Time']


    if first_processing < first_wash:

        start_time = first_processing
    else:
        start_time = first_wash

    return (start_time , first_processing, first_wash)

def find_tasks(df): 

    tasks = [] 

    for i, row in df.iterrows():
        
        product = row['product name']
        product_qty = row['quantity liters']
        process_speed = row['process speed per hour']
        line_efficiency = row['line efficiency']
        hourly_speed = process_speed * line_efficiency
        process_duration = (hourly_speed / product_qty) * 60
        change_over = row['Change Over']
        first_wash_duration = row['Duration']
        duration_gap = row['Gap']
        int_wash_duration = row['Intermediate Wash Duration']
        additional_wash = row['Additional Wash']
        task = (product_name,process_duration,change_over,first_wash_duration,duration_gap,int_wash_duration,additional_wash)
        tasks.append(task)

    return tasks

def full_wash(begin,duration,wash_type= 'scheduled'):
    


    
df = pd.read_excel('updated_example_production_data with wash time - Copy 2.xlsx') #change to uploaded file 
print(starting_time(df))
print(find_tasks(df))
#print(find_tasks(df)) 
print(df.columns.tolist())
for row in df:
    print(f'row value: {row}')

for i, row in df.iterrows():
    product_name = row['product name']
    print(product_name)
