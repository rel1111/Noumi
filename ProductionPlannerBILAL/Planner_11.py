import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import streamlit as st
import io


def generate_timeline(df):
    """
    Clean rebuild of production timeline generator with simplified 24hr wash logic.
    """
    
    # Define colors
    colors = {
        'processing': 'darkgreen',
        'wash': 'purple',
        'changeover': 'darkorange', 
        'intermediate_wash': 'lightblue'
    }
    
    tasks = []
    
    # Parse parameters from first row
    try:
        start_time = pd.to_datetime(df.loc[0, 'Date from'])
        wash_duration_mins = int(df.loc[0, 'Duration']) if pd.notna(df.loc[0, 'Duration']) else 0
        wash_gap_mins = int(df.loc[0, 'Gap']) if pd.notna(df.loc[0, 'Gap']) else 0
        wash_duration = timedelta(minutes=wash_duration_mins)
        gap_duration = timedelta(minutes=wash_gap_mins)
        
        first_wash_time = None
        if pd.notna(df.loc[0, 'First Wash Time']):
            first_wash_time = pd.to_datetime(df.loc[0, 'First Wash Time'])
            
    except Exception as e:
        st.error(f"Error parsing schedule parameters: {e}")
        return None
    
    # Constants
    intermediate_duration = timedelta(minutes=180)  # 3 hours
    st.info("Intermediate wash duration: 180 minutes (3 hours)")
    
    # Tracking variables
    current_time = start_time
    last_wash_end_time = first_wash_time if first_wash_time else start_time
    last_wash_of_any_kind = None  # Simple tracker for 24hr rule
    
    # Add first wash if specified
    if first_wash_time and wash_duration > timedelta(0):
        tasks.extend([
            {
                'start': first_wash_time,
                'end': first_wash_time + wash_duration,
                'task': 'wash',
                'product': 'Scheduled Wash',
                'order': -2
            },
            {
                'start': first_wash_time,
                'end': first_wash_time + wash_duration,
                'task': 'intermediate_wash', 
                'product': 'Intermediate Wash',
                'order': -1
            }
        ])
        last_wash_end_time = first_wash_time + wash_duration
        last_wash_of_any_kind = first_wash_time + wash_duration
    
    # Process each product
    for i, row in df.iterrows():
        product_name = row['product name']
        quantity_liters = row['quantity liters']
        process_speed = row['process speed per hour'] 
        line_efficiency = row['line efficiency']
        change_over_mins = row['Change Over']
        additional_wash = row.get('Additional Wash', 'No')
        
        # 1. Additional wash if needed
        if additional_wash == 'Yes' and wash_duration > timedelta(0):
            tasks.extend([
                {
                    'start': current_time,
                    'end': current_time + wash_duration,
                    'task': 'wash',
                    'product': 'Scheduled Wash',
                    'order': -2
                },
                {
                    'start': current_time,
                    'end': current_time + wash_duration,
                    'task': 'intermediate_wash',
                    'product': 'Intermediate Wash', 
                    'order': -1
                }
            ])
            current_time += wash_duration
            last_wash_end_time = current_time
            last_wash_of_any_kind = current_time
        
        # 2. Changeover (skip first product)
        if i > 0 and change_over_mins > 0:
            changeover_duration = timedelta(minutes=change_over_mins)
            
            # Check for scheduled wash overlap
            next_wash_time = last_wash_end_time + gap_duration
            changeover_end = current_time + changeover_duration
            
            if (wash_duration > timedelta(0) and 
                next_wash_time < changeover_end and 
                next_wash_time + wash_duration > current_time):
                
                # Scheduled wash overlaps - skip changeover, use wash
                tasks.extend([
                    {
                        'start': next_wash_time,
                        'end': next_wash_time + wash_duration,
                        'task': 'wash',
                        'product': 'Scheduled Wash',
                        'order': -2
                    },
                    {
                        'start': next_wash_time,
                        'end': next_wash_time + wash_duration,
                        'task': 'intermediate_wash',
                        'product': 'Intermediate Wash',
                        'order': -1
                    }
                ])
                current_time = next_wash_time + wash_duration
                last_wash_end_time = current_time
                last_wash_of_any_kind = current_time
                
            else:
                # Check if changeover can overlap with an additional wash that just completed
                # Look for recent additional wash that could accommodate this changeover
                can_overlap_with_recent_wash = False
                for task in tasks:
                    if (task['task'] == 'wash' and 
                        task['product'] == 'Scheduled Wash' and
                        task['end'] == current_time and  # Wash just ended
                        changeover_duration <= wash_duration):  # Changeover fits in wash duration
                        
                        # Move changeover to overlap with the end of this wash
                        changeover_start = task['end'] - changeover_duration
                        tasks.append({
                            'start': changeover_start,
                            'end': task['end'],
                            'task': 'changeover',
                            'product': product_name,
                            'order': i
                        })
                        can_overlap_with_recent_wash = True
                        break
                
                if not can_overlap_with_recent_wash:
                    # Normal changeover
                    tasks.append({
                        'start': current_time,
                        'end': changeover_end,
                        'task': 'changeover',
                        'product': product_name,
                        'order': i
                    })
                    current_time = changeover_end
        
        # 3. Check for 24hr standalone intermediate wash BEFORE processing
        needs_standalone = False
        if last_wash_of_any_kind:
            hours_since_wash = (current_time - last_wash_of_any_kind).total_seconds() / 3600
            if hours_since_wash >= 24:
                needs_standalone = True
        elif not last_wash_of_any_kind and current_time > start_time + timedelta(hours=24):
            # No wash ever, but 24+ hours since start
            needs_standalone = True
            
        if needs_standalone:
            tasks.append({
                'start': current_time,
                'end': current_time + intermediate_duration,
                'task': 'intermediate_wash',
                'product': 'Intermediate Wash',
                'order': -1
            })
            current_time += intermediate_duration
            last_wash_of_any_kind = current_time
        
        # 4. Processing
        effective_speed = process_speed * line_efficiency
        processing_hours = quantity_liters / effective_speed
        processing_start = current_time
        processing_end = current_time + timedelta(hours=processing_hours)
        
        # Find scheduled washes during processing
        wash_interruptions = []
        if wash_duration > timedelta(0):
            next_wash = last_wash_end_time + gap_duration
            while next_wash < processing_end:
                wash_end = next_wash + wash_duration
                wash_interruptions.append((next_wash, wash_end))
                next_wash = wash_end + gap_duration
        
        # Extend processing for wash interruptions
        total_wash_time = sum(
            (wash_end - wash_start).total_seconds() / 3600
            for wash_start, wash_end in wash_interruptions
            if wash_start >= processing_start
        )
        extended_processing_end = processing_end + timedelta(hours=total_wash_time)
        
        # Add wash tasks that occur during processing
        for wash_start, wash_end in wash_interruptions:
            if wash_start >= processing_start:
                tasks.extend([
                    {
                        'start': wash_start,
                        'end': wash_end,
                        'task': 'wash',
                        'product': 'Scheduled Wash',
                        'order': -2
                    },
                    {
                        'start': wash_start,
                        'end': wash_end,
                        'task': 'intermediate_wash',
                        'product': 'Intermediate Wash',
                        'order': -1
                    }
                ])
                last_wash_end_time = wash_end
                last_wash_of_any_kind = wash_end
        
        # Create processing segments around washes
        segment_start = processing_start
        for wash_start, wash_end in wash_interruptions:
            if wash_start >= processing_start:
                if segment_start < wash_start:
                    tasks.append({
                        'start': segment_start,
                        'end': wash_start,
                        'task': 'processing',
                        'product': product_name,
                        'order': i
                    })
                segment_start = wash_end
        
        # Final processing segment
        if segment_start < extended_processing_end:
            tasks.append({
                'start': segment_start,
                'end': extended_processing_end,
                'task': 'processing',
                'product': product_name,
                'order': i
            })
        
        # Update current time
        current_time = extended_processing_end
    
    # Create visualization
    if not tasks:
        st.error("No tasks generated. Please check your data.")
        return None
    
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_facecolor('white')
    
    # Organize products for display
    tasks_df = pd.DataFrame(tasks)
    wash_products = ['Scheduled Wash', 'Intermediate Wash']
    other_products = [p for p in tasks_df['product'].unique() if p not in wash_products]
    product_order = [p for p in wash_products if p in tasks_df['product'].unique()] + other_products
    
    # Plot tasks
    y_pos = 0
    for product_name in product_order:
        product_tasks = tasks_df[tasks_df['product'] == product_name].sort_values('start')
        
        for _, task in product_tasks.iterrows():
            duration = task['end'] - task['start']
            ax.broken_barh(
                [(task['start'], duration)],
                (y_pos - 0.4, 0.8),
                facecolors=colors[task['task']],
                edgecolor='black'
            )
            
            # Add timestamps for wash events
            if task['task'] in ['wash', 'intermediate_wash']:
                ax.vlines(task['start'], ymin=-0.5, ymax=y_pos + 0.4,
                         color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.vlines(task['end'], ymin=-0.5, ymax=y_pos + 0.4,
                         color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Timestamp labels with yellow outline
                ax.text(task['start'], -0.1, task['start'].strftime('%H:%M'),
                       rotation=90, va='top', ha='right', fontsize=8, fontweight='bold',
                       color='black', path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
                ax.text(task['end'], -0.1, task['end'].strftime('%H:%M'),
                       rotation=90, va='top', ha='right', fontsize=8, fontweight='bold',
                       color='black', path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
            
            # Add timestamps for processing (debugging)
            if task['task'] == 'processing':
                ax.text(task['start'], y_pos + 0.5, task['start'].strftime('%H:%M'),
                       rotation=90, va='bottom', ha='left', fontsize=7, fontweight='bold',
                       color='black', path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])
                ax.text(task['end'], y_pos + 0.5, task['end'].strftime('%H:%M'),
                       rotation=90, va='bottom', ha='left', fontsize=7, fontweight='bold',
                       color='black', path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])
        
        y_pos += 1
    
    # Configure plot
    ax.set_yticks(range(len(product_order)))
    ax.set_yticklabels(product_order)
    ax.set_xlabel("Time")
    ax.set_title("Weekly Production Plan Timeline")
    ax.grid(True)
    ax.invert_yaxis()
    
    # Format time axis
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    
    # Add day division lines
    first_date = tasks_df['start'].min().floor('D')
    last_date = tasks_df['end'].max().ceil('D')
    delta_days = (last_date - first_date).days + 1
    for day in range(delta_days):
        day_start = first_date + timedelta(days=day)
        ax.axvline(day_start, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
    plt.xticks(rotation=90, ha='right', va='top')
    plt.setp(ax.get_xminorticklabels(), rotation=90, ha='right', va='top')
    
    # Legend
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
            optional_columns = ['First Wash Time', 'Additional Wash']
            
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
            st.write("**Optional:** First Wash Time, Additional Wash")
            st.write("**Data Format:** All wash parameters (Duration, Gap, First Wash Time) should be in the first row.")


if __name__ == "__main__":
    main()