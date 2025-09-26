import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import streamlit as st
import io


def generate_timeline(df):
    """
    Processes the production plan data and generates a timeline.

    Args:
        df (pd.DataFrame): The DataFrame containing the production plan data.

    Returns:
        matplotlib.figure.Figure: The generated timeline figure.
    """

    # Define colors for each task
    colors = {
        'processing': 'darkgreen',
        'wash': 'purple',
        'changeover': 'darkorange',
        'intermediate_wash': 'lightblue'
    }

    tasks = []

    # ----------------------------
    # Helper functions
    # ----------------------------
    def add_intermediate_wash(start_time, end_time, duration):
        """Helper to add intermediate wash task"""
        return {
            'start': start_time,
            'end': end_time,
            'duration_hours': duration.total_seconds() / 3600,
            'task': 'intermediate_wash',
            'product': 'Intermediate Wash',
            'order': -0.5
        }

    def needs_24hr_intermediate_wash(current_processing_time, last_intermediate_time):
        """Check if we need a 24hr standalone intermediate wash based on processing time"""
        if last_intermediate_time is None:
            return False
        # Check if 24 hours have passed since last intermediate wash from the perspective of processing
        return (current_processing_time - last_intermediate_time) >= timedelta(hours=24)

    def find_next_intermediate_wash_time(current_processing_start, current_processing_end, last_processing_start_time, last_intermediate_wash_time):
        """Find when the next intermediate wash should occur - 24hrs from processing start"""
        if last_processing_start_time is None:
            return None
        
        # Calculate when 24 hours will have passed since PROCESSING started
        next_intermediate_due = last_processing_start_time + timedelta(hours=24)
        
        # Only schedule if:
        # 1. The 24hr mark falls within current processing period
        # 2. We haven't already done an intermediate wash recently (from scheduled washes)
        if (next_intermediate_due >= current_processing_start and 
            next_intermediate_due < current_processing_end):
            
            # Check if we already had a recent intermediate wash from scheduled washes
            if last_intermediate_wash_time and (next_intermediate_due - last_intermediate_wash_time) < timedelta(hours=24):
                return None  # Skip because we already had a recent intermediate wash
            
            return next_intermediate_due
        return None

    # ----------------------------
    # Parse schedule parameters
    # ----------------------------
    try:
        start_time_of_week = pd.to_datetime(df.loc[0, 'Date from'])
    except Exception as e:
        st.error(f"Error parsing 'Date from' column: {e}. Please ensure it's in a valid date/time format.")
        return None

    current_time = start_time_of_week

    try:
        wash_duration_mins = int(df.loc[0, 'Duration'])
        wash_gap_mins = int(df.loc[0, 'Gap'])
        wash_duration = timedelta(minutes=wash_duration_mins)
        gap_duration = timedelta(minutes=wash_gap_mins)
    except Exception as e:
        st.warning(f"Warning: Error reading wash duration/gap: {e}. Using 0.")
        wash_duration = timedelta(minutes=0)
        gap_duration = timedelta(minutes=0)

    try:
        if 'Intermediate Wash Duration' in df.columns and pd.notna(df.loc[0, 'Intermediate Wash Duration']):
            intermediate_duration_mins = int(df.loc[0, 'Intermediate Wash Duration'])
        else:
            intermediate_duration_mins = 180
        intermediate_wash_duration = timedelta(minutes=intermediate_duration_mins)
        st.info(f"Intermediate wash duration set to {intermediate_duration_mins} minutes")
    except Exception:
        intermediate_wash_duration = timedelta(minutes=180)

    first_wash_time = None
    try:
        first_wash_time = pd.to_datetime(df.loc[0, 'First Wash Time'])
        last_wash_end_time = first_wash_time
    except Exception:
        last_wash_end_time = start_time_of_week

    last_intermediate_wash_time = None
    last_processing_start_time = None  # Track when processing last started
    processing_start_times = []

    # ----------------------------
    # First wash if specified
    # ----------------------------
    if first_wash_time and wash_duration > timedelta(minutes=0):
        # Scheduled wash
        tasks.append({
            'start': first_wash_time,
            'end': first_wash_time + wash_duration,
            'duration_hours': wash_duration.total_seconds() / 3600,
            'task': 'wash',
            'product': 'Scheduled Wash',
            'order': -2
        })
        # Simultaneous intermediate wash
        tasks.append(add_intermediate_wash(first_wash_time, first_wash_time + wash_duration, wash_duration))
        last_intermediate_wash_time = first_wash_time + wash_duration
        last_wash_end_time = first_wash_time + wash_duration

    # ----------------------------
    # Loop over products
    # ----------------------------
    for i, row in df.iterrows():
        product_name = row['product name']
        quantity_liters = row['quantity liters']
        process_speed = row['process speed per hour']
        line_efficiency = row['line efficiency']
        change_over_mins = row['Change Over']

        # --- Changeover ---
        if i > 0:
            change_over_duration = timedelta(minutes=change_over_mins)
            changeover_end_time = current_time + change_over_duration

            next_wash_start_time = last_wash_end_time + gap_duration
            scheduled_washes_in_changeover = []
            while next_wash_start_time < changeover_end_time:
                wash_end_time = next_wash_start_time + wash_duration
                scheduled_washes_in_changeover.append({
                    'start': next_wash_start_time,
                    'end': wash_end_time
                })
                last_wash_end_time = wash_end_time
                next_wash_start_time = last_wash_end_time + gap_duration

            changeover_overlaps_with_wash = any(
                max(current_time, wash['start']) < min(changeover_end_time, wash['end'])
                for wash in scheduled_washes_in_changeover
            )

            if changeover_overlaps_with_wash:
                for wash in scheduled_washes_in_changeover:
                    # Scheduled wash
                    tasks.append({
                        'start': wash['start'],
                        'end': wash['end'],
                        'duration_hours': wash_duration.total_seconds() / 3600,
                        'task': 'wash',
                        'product': 'Scheduled Wash',
                        'order': -1
                    })
                    # Simultaneous intermediate wash
                    tasks.append(add_intermediate_wash(wash['start'], wash['end'], wash_duration))
                    last_intermediate_wash_time = wash['end']
                current_time = scheduled_washes_in_changeover[-1]['end']
            else:
                tasks.append({
                    'start': current_time,
                    'end': changeover_end_time,
                    'duration_hours': change_over_mins / 60,
                    'task': 'changeover',
                    'product': product_name,
                    'order': i
                })
                current_time = changeover_end_time

        # --- Processing ---
        effective_speed = process_speed * line_efficiency
        total_processing_hours = quantity_liters / effective_speed
        processing_end_time = current_time + timedelta(hours=total_processing_hours)
        processing_start_time = current_time  # Store the actual processing start time

        total_wash_overlap_duration = timedelta(minutes=0)
        next_wash_start_time = last_wash_end_time + gap_duration

        # Handle scheduled washes during processing
        while next_wash_start_time < processing_end_time + total_wash_overlap_duration:
            wash_end_time = next_wash_start_time + wash_duration
            overlap_start = max(current_time, next_wash_start_time)
            overlap_end = min(processing_end_time + total_wash_overlap_duration, wash_end_time)

            if overlap_start < overlap_end:
                total_wash_overlap_duration += (overlap_end - overlap_start)

                # Scheduled wash
                tasks.append({
                    'start': next_wash_start_time,
                    'end': wash_end_time,
                    'duration_hours': wash_duration.total_seconds() / 3600,
                    'task': 'wash',
                    'product': 'Scheduled Wash',
                    'order': -1
                })
                # Simultaneous intermediate wash
                tasks.append(add_intermediate_wash(next_wash_start_time, wash_end_time, wash_duration))
                last_intermediate_wash_time = wash_end_time

                last_wash_end_time = wash_end_time
                next_wash_start_time = last_wash_end_time + gap_duration
            else:
                break

        extended_processing_end_time = processing_end_time + total_wash_overlap_duration
        processing_start_times.append(current_time)


        # --- Standalone 24hr Intermediate Wash Check ---
        if last_intermediate_wash_time is not None:
            intermediate_due_time = last_intermediate_wash_time + timedelta(hours=24)

            if processing_start_time <= intermediate_due_time <= extended_processing_end_time:
                # Avoid duplicates if we already have a wash at/near this time
                wash_tolerance = timedelta(hours=1)  # adjustable window
                overlap = any(
                    abs((t['start'] - intermediate_due_time)) <= wash_tolerance
                    for t in tasks if t['task'] == 'intermediate_wash'
                )

                if not overlap:
                    tasks.append(add_intermediate_wash(
                        intermediate_due_time,
                        intermediate_due_time + intermediate_wash_duration,
                        intermediate_wash_duration
                    ))
                    last_intermediate_wash_time = intermediate_due_time + intermediate_wash_duration

                # Always extend processing time by the full wash duration
                    extended_processing_end_time += intermediate_wash_duration

        
        # Update the last processing start time for next iteration
        last_processing_start_time = processing_start_time

        # Create processing segments around wash intervals
        segment_start_time = current_time
        wash_intervals = [(t['start'], t['end']) for t in tasks if t['task'] in ['wash', 'intermediate_wash']]
        wash_intervals = sorted([w for w in wash_intervals if w[0] < extended_processing_end_time and w[1] > current_time])

        for wash_start, wash_end in wash_intervals:
            if segment_start_time < wash_start:
                tasks.append({
                    'start': segment_start_time,
                    'end': wash_start,
                    'duration_hours': (wash_start - segment_start_time).total_seconds() / 3600,
                    'task': 'processing',
                    'product': product_name,
                    'order': i
                })
            segment_start_time = max(segment_start_time, wash_end)

        if segment_start_time < extended_processing_end_time:
            tasks.append({
                'start': segment_start_time,
                'end': extended_processing_end_time,
                'duration_hours': (extended_processing_end_time - segment_start_time).total_seconds() / 3600,
                'task': 'processing',
                'product': product_name,
                'order': i
            })

        current_time = extended_processing_end_time
        
    # ----------------------------
    # Plotting
    # ----------------------------
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_facecolor('white')

    y_pos = 0
    product_y_positions = {}
    product_order = []

    tasks_df = pd.DataFrame(tasks).sort_values(by='order')

    wash_products = ['Scheduled Wash', 'Intermediate Wash']
    other_products = [p for p in tasks_df['product'].unique() if p not in wash_products]
    unique_products_ordered = [p for p in wash_products if p in tasks_df['product'].unique()] + other_products

    for product_name in unique_products_ordered:
        group = tasks_df[tasks_df['product'] == product_name].sort_values(by='start')
        product_y_positions[product_name] = y_pos
        product_order.append(product_name)

        for _, task in group.iterrows():
            ax.broken_barh(
                [(task['start'], task['end'] - task['start'])],
                (y_pos - 0.4, 0.8),
                facecolors=colors[task['task']],
                edgecolor='black'
            )

            # Draw vertical lines only for washes
            if task['task'] in ['wash', 'intermediate_wash']:
                ax.vlines(task['start'], ymin=-0.5, ymax=y_pos + 0.4,
                          color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.vlines(task['end'], ymin=-0.5, ymax=y_pos + 0.4,
                          color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

                # Add start and end time labels only for washes - black text with yellow outline
                ax.text(task['start'], -0.1, task['start'].strftime('%H:%M'),
                        rotation=90, va='top', ha='right',
                        fontsize=8, fontweight='bold', color='black',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
                ax.text(task['end'], -0.1, task['end'].strftime('%H:%M'),
                        rotation=90, va='top', ha='right',
                        fontsize=8, fontweight='bold', color='black',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
            
            if task['task'] == 'processing':
                # Add start and end time labels for processing - black text with green outline
                ax.text(task['start'], y_pos + 0.5, task['start'].strftime('%H:%M'),
                        rotation=90, va='bottom', ha='left',
                        fontsize=7, fontweight='bold', color='black',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])
                ax.text(task['end'], y_pos + 0.5, task['end'].strftime('%H:%M'),
                        rotation=90, va='bottom', ha='left',
                        fontsize=7, fontweight='bold', color='black',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])

        y_pos += 1

    ax.set_yticks(list(product_y_positions.values()))
    ax.set_yticklabels(product_order)
    ax.set_xlabel("Time")
    ax.set_title("Weekly Production Plan Timeline")
    ax.grid(True)
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

    if not tasks_df.empty:
        first_date = tasks_df['start'].min().floor('D')
        last_date = tasks_df['end'].max().ceil('D')
        delta_days = (last_date - first_date).days + 1
        for day in range(delta_days):
            day_start = first_date + timedelta(days=day)
            ax.axvline(day_start, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    plt.xticks(rotation=90, ha='right', va='top')
    plt.setp(ax.get_xminorticklabels(), rotation=90, ha='right', va='top')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[t]) for t in colors]
    ax.legend(handles, colors.keys(), loc='upper right')

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Production Plan Timeline Generator", layout="wide")

    st.title("Production Plan Timeline Generator")
    st.write("Upload your production plan file (Excel or CSV) to generate a timeline visualization.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload an Excel or CSV file containing your production plan data."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"File uploaded successfully: {uploaded_file.name}")

            st.subheader("Data Preview")
            st.dataframe(df.head())

            required_columns = [
                'product name', 'quantity liters', 'process speed per hour',
                'line efficiency', 'Change Over', 'Date from', 'Duration', 'Gap'
            ]
            optional_columns = ['First Wash Time', 'Intermediate Wash Duration']

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write("**Required columns:**")
                for col in required_columns:
                    status = "✅" if col in df.columns else "❌"
                    st.write(f"{status} {col}")
                st.write("**Optional columns:**")
                for col in optional_columns:
                    status = "✅" if col in df.columns else "⚪"
                    st.write(f"{status} {col}")
            else:
                st.success("All required columns found!")

                for col in optional_columns:
                    if col in df.columns:
                        st.info(f"Optional column '{col}' found - will be used.")
                    else:
                        st.info(f"Optional column '{col}' not found - defaults will be used.")

                if st.button("Generate Timeline", type="primary"):
                    with st.spinner("Generating timeline..."):
                        fig = generate_timeline(df)
                        if fig:
                            st.subheader("Production Timeline")
                            st.pyplot(fig)

                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                            buf.seek(0)

                            st.download_button(
                                label="Download Timeline as PNG",
                                data=buf.getvalue(),
                                file_name="production_timeline.png",
                                mime="image/png"
                            )
                        else:
                            st.error("Failed to generate timeline. Please check your data format.")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("Please upload a file to get started.")

        with st.expander("Expected File Format"):
            st.write("Your file should contain these columns:")
            st.write("**Required:** product name, quantity liters, process speed per hour, line efficiency, Change Over, Date from, Duration, Gap")
            st.write("**Optional:** First Wash Time, Intermediate Wash Duration")


if __name__ == "__main__":
    main()