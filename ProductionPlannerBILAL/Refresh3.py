import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def parse_input(df):
    # Assumes columns (loosely named): product name, quantity liters, process speed per hour, line efficiency,
    # changeover, date from, duration, gap, first wash time, additional wash
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def calculate_processing_time(row):
    # Processing time in hours = quantity / (speed * efficiency)
    return row['quantity_liters'] / (row['process_speed_per_hour'] * row['line_efficiency'])

def add_time(dt, minutes):
    return dt + timedelta(minutes=minutes)

def generate_timeline(df, wash_params):
    """
    Generate a timeline dataframe with all events: processing, washes, changeovers, intermediate washes.
    wash_params includes:
     - scheduled_duration (min)
     - scheduled_gap (min)
     - first_wash_time (datetime)
     - intermediate_duration = 180 (min, fixed)
    """

    timeline = []
    current_time = None
    scheduled_interval = wash_params['scheduled_duration'] + wash_params['scheduled_gap']
    intermediate_duration = 180

    # Helper to schedule intermediate wash simultaneously
    def schedule_wash(start, duration, event_type, color):
        timeline.append(dict(Task=event_type, Start=start, Finish=start + timedelta(minutes=duration), Color=color))
        # intermediate wash always simultaneous except standalone
        if event_type in ['Scheduled Wash', 'Additional Wash']:
            timeline.append(dict(Task='Intermediate Wash', Start=start,
                                 Finish=start + timedelta(minutes=intermediate_duration), Color='lightblue'))

    # Initialize scheduled wash timing
    next_scheduled_wash = wash_params['first_wash_time']

    for i, row in df.iterrows():
        processing_hours = calculate_processing_time(row)
        if i == 0:
            current_time = row['date_from']
        else:
            # Changeover logic
            changeover_start = current_time
            changeover_end = add_time(changeover_start, row['change_over'])
            # Check if changeover overlaps any scheduled wash
            if next_scheduled_wash and changeover_start <= next_scheduled_wash <= changeover_end:
                # skip changeover, use wash time instead (wash will appear anyway)
                pass
            else:
                # add changeover block
                timeline.append(dict(Task=f"Changeover before {row['product_name']}", Start=changeover_start, Finish=changeover_end, Color='orange'))
            current_time = changeover_end

        processing_end = current_time + timedelta(hours=processing_hours)
        wash_events = []

        # Additional wash before processing product if flagged "Yes"
        if str(row['additional_wash']).strip().lower() == "yes":
            schedule_wash(current_time, wash_params['scheduled_duration'], 'Additional Wash', 'purple')
            current_time += timedelta(minutes=wash_params['scheduled_duration'])

        # Scheduled washes during processing
        # For simplicity, we simulate washes that interrupt processing at fixed intervals from first wash time
        # Calculate interrupts and extend processing by wash duration each time
        process_start = current_time
        process_end = process_start + timedelta(hours=processing_hours)

        wash_interrupts = []
        # Generate scheduled wash times within processing interval
        wash_time = next_scheduled_wash
        while wash_time and wash_time < process_end:
            if wash_time > process_start:
                wash_interrupts.append(wash_time)
            wash_time += timedelta(minutes=scheduled_interval)

        # Extend processing by washes durations
        total_extension = len(wash_interrupts) * wash_params['scheduled_duration']
        processing_end += timedelta(minutes=total_extension)

        # Add processing segments and washes in timeline
        segment_start = process_start
        for w_start in wash_interrupts:
            # Processing segment before wash
            timeline.append(dict(Task=row['product_name'], Start=segment_start, Finish=w_start, Color='green'))
            # Schedule wash + intermediate wash
            schedule_wash(w_start, wash_params['scheduled_duration'], 'Scheduled Wash', 'purple')
            segment_start = w_start + timedelta(minutes=wash_params['scheduled_duration'])

        # Final segment after last wash till processing end
        timeline.append(dict(Task=row['product_name'], Start=segment_start, Finish=processing_end, Color='green'))

        current_time = processing_end
        next_scheduled_wash = wash_interrupts[-1] + timedelta(minutes=scheduled_interval) if wash_interrupts else next_scheduled_wash

        # Standalone intermediate wash if 24hrs processing with no wash
        # (For simplification, not implemented here fully)

    timeline_df = pd.DataFrame(timeline)
    return timeline_df

st.title("Production Timeline Generator")

uploaded_file = st.file_uploader("Upload Excel with production schedule", type=['xls', 'xlsx'])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = parse_input(df)

    # You can replace below with actual values parsed from Excel if available
    wash_params = {
        'scheduled_duration': 360, # 6 hours in minutes
        'scheduled_gap': 3120,     # gap in minutes as example
        'first_wash_time': datetime.strptime('2025-09-21 22:30:00', '%Y-%m-%d %H:%M:%S'),
        'intermediate_duration': 180
    }

    # Parse datetime columns if not already
    for col in ['date_from', 'first_wash_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    timeline_df = generate_timeline(df, wash_params)

    # Plot Gantt chart with color coding
    fig = px.timeline(timeline_df, x_start="Start", x_end="Finish", y="Task", color="Color",
                      color_discrete_map={
                          'green': 'green',
                          'purple': 'purple',
                          'orange': 'orange',
                          'lightblue': 'lightblue',
                          None: 'grey'
                      })
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Detailed Timeline Data", timeline_df)
