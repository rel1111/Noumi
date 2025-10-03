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
    
    Updates:
    - Removes rows with empty/NaN product names before processing
    - New column 'Qty of batch' (default 30k if empty)
    - Batch markers (red dashed lines) for each batch
    """
    
    # === FIX: Clean the dataframe first ===
    df = df.dropna(subset=['product name'])
    df = df[df['product name'].str.strip() != '']
    df = df.reset_index(drop=True)
    
    st.info(f"Cleaned data: {len(df)} valid product rows found")
    
    # === Add Qty of batch column ===
    def compute_batches(qty):
        if pd.isna(qty) or qty == '':
            qty = 30000  # default
        try:
            qty = int(str(qty).replace(',', ''))
        except:
            qty = 30000
        if qty <= 30000:
            return 1
        return -(-qty // 30000)  # ceiling division
    
    df['Qty of batch'] = df['quantity liters'].apply(compute_batches)
    
    # Helper function to safely parse numbers
    def safe_number_parse(value, default=0):
        try:
            if pd.isna(value) or value == '':
                return default
            if isinstance(value, str):
                clean_value = value.replace(',', '')
                return int(float(clean_value))
            return int(float(value))
        except (ValueError, TypeError):
            return default

    # colors
    colors = {
        'processing': 'darkgreen',
        'wash': 'purple',
        'changeover': 'darkorange',
        'intermediate_wash': 'lightblue',
        'batch_marker': 'red'
    }

    tasks = []

    # Parse schedule parameters
    try:
        start_time = pd.to_datetime(df.loc[0, 'Date from'])
        wash_duration_mins = safe_number_parse(df.loc[0, 'Duration'], 0)
        wash_gap_mins = safe_number_parse(df.loc[0, 'Gap'], 0)
        wash_duration = timedelta(minutes=wash_duration_mins)
        gap_duration = timedelta(minutes=wash_gap_mins)

        first_wash_time = None
        if 'First Wash Time' in df.columns and pd.notna(df.loc[0, 'First Wash Time']):
            try:
                first_wash_time = pd.to_datetime(df.loc[0, 'First Wash Time'])
            except (ValueError, TypeError):
                first_wash_time = None
    except Exception as e:
        st.error(f"Error parsing schedule parameters: {e}")
        return None

    intermediate_duration = timedelta(minutes=180)
    st.info(f"Intermediate wash duration: 180 minutes (3 hours)")

    current_time = start_time
    last_wash_end_time = first_wash_time if first_wash_time else start_time
    last_processing_start_after_wash = None
    reset_24h_on_next_processing = False

    if first_wash_time and wash_duration > timedelta(0):
        next_scheduled_wash = first_wash_time + wash_duration + gap_duration
    elif wash_duration > timedelta(0):
        next_scheduled_wash = start_time + gap_duration
    else:
        next_scheduled_wash = None

    # First wash if specified
    if first_wash_time and wash_duration > timedelta(0):
        tasks.append({
            'start': first_wash_time,
            'end': first_wash_time + wash_duration,
            'task': 'wash',
            'product': 'Scheduled Wash',
            'order': -2
        })
        tasks.append({
            'start': first_wash_time,
            'end': first_wash_time + wash_duration,
            'task': 'intermediate_wash',
            'product': 'Intermediate Wash',
            'order': -1
        })
        last_wash_end_time = first_wash_time + wash_duration
        if first_wash_time + wash_duration > current_time:
            current_time = first_wash_time + wash_duration + timedelta(minutes=1)
        reset_24h_on_next_processing = True

    def total_extension_hours(wash_list):
        return sum((w['end'] - w['start']).total_seconds() / 3600.0 for w in wash_list)

    # Process products
    for i, row in df.iterrows():
        product_name = row['product name']
        quantity_liters = safe_number_parse(row['quantity liters'], 0)
        process_speed = safe_number_parse(row['process speed per hour'], 1)
        line_efficiency = float(str(row['line efficiency']).replace(',', '')) if pd.notna(row['line efficiency']) else 0.0
        change_over_mins = safe_number_parse(row['Change Over'], 0)
        additional_wash = row.get('Additional Wash', 'No') if 'Additional Wash' in row else 'No'

        # Insert any scheduled washes that must occur before this product's start time
        while next_scheduled_wash and next_scheduled_wash <= current_time:
            wash_end = next_scheduled_wash + wash_duration
            tasks.append({
                'start': next_scheduled_wash,
                'end': wash_end,
                'task': 'wash',
                'product': 'Scheduled Wash',
                'order': -2
            })
            tasks.append({
                'start': next_scheduled_wash,
                'end': wash_end,
                'task': 'intermediate_wash',
                'product': 'Intermediate Wash',
                'order': -1
            })
            last_wash_end_time = wash_end
            current_time = max(current_time, wash_end + timedelta(minutes=1))
            reset_24h_on_next_processing = True
            next_scheduled_wash = wash_end + gap_duration

        # Additional wash before product (if requested)
        additional_wash_end = None
        if additional_wash == 'Yes' and wash_duration > timedelta(0):
            additional_wash_end = current_time + wash_duration
            tasks.append({
                'start': current_time,
                'end': additional_wash_end,
                'task': 'wash',
                'product': 'Scheduled Wash',
                'order': -2
            })
            tasks.append({
                'start': current_time,
                'end': additional_wash_end,
                'task': 'intermediate_wash',
                'product': 'Intermediate Wash',
                'order': -1
            })
            current_time = additional_wash_end + timedelta(minutes=1)
            last_wash_end_time = additional_wash_end
            reset_24h_on_next_processing = True

        # Changeover (skip for first product)
        if i > 0:
            changeover_duration = timedelta(minutes=change_over_mins)
            changeover_end = current_time + changeover_duration

            if next_scheduled_wash and next_scheduled_wash < changeover_end and (next_scheduled_wash + wash_duration) > current_time:
                wash_end = next_scheduled_wash + wash_duration
                tasks.append({
                    'start': next_scheduled_wash,
                    'end': wash_end,
                    'task': 'wash',
                    'product': 'Scheduled Wash',
                    'order': -2
                })
                tasks.append({
                    'start': next_scheduled_wash,
                    'end': wash_end,
                    'task': 'intermediate_wash',
                    'product': 'Intermediate Wash',
                    'order': -1
                })
                current_time = wash_end + timedelta(minutes=1)
                last_wash_end_time = wash_end
                reset_24h_on_next_processing = True
                next_scheduled_wash = wash_end + gap_duration
            elif additional_wash_end and changeover_duration <= wash_duration:
                changeover_start = additional_wash_end - changeover_duration
                tasks.append({
                    'start': changeover_start,
                    'end': additional_wash_end,
                    'task': 'changeover',
                    'product': product_name,
                    'order': i
                })
            else:
                tasks.append({
                    'start': current_time,
                    'end': changeover_end,
                    'task': 'changeover',
                    'product': product_name,
                    'order': i
                })
                current_time = changeover_end

        # Processing calculation
        effective_speed = process_speed * line_efficiency
        if effective_speed == 0:
            st.error(f"Effective speed for product '{product_name}' is zero.")
            return None

        processing_hours = quantity_liters / effective_speed
        processing_start = current_time
        processing_end = current_time + timedelta(hours=processing_hours)

        if reset_24h_on_next_processing:
            last_processing_start_after_wash = processing_start
            reset_24h_on_next_processing = False
        elif last_processing_start_after_wash is None and i == 0:
            last_processing_start_after_wash = processing_start

        # Build wash interruptions
        wash_interruptions = []

        while True:
            added_any = False
            window_end = processing_end + timedelta(hours=total_extension_hours(wash_interruptions))

            while next_scheduled_wash and next_scheduled_wash < window_end:
                wash_end = next_scheduled_wash + wash_duration
                wash_interruptions.append({
                    'start': next_scheduled_wash,
                    'end': wash_end,
                    'type': 'scheduled'
                })
                next_scheduled_wash = wash_end + gap_duration
                added_any = True

            window_end = processing_end + timedelta(hours=total_extension_hours(wash_interruptions))

            if last_processing_start_after_wash:
                standalone_due = last_processing_start_after_wash + timedelta(hours=24)
                
                if processing_start <= standalone_due <= window_end:
                    has_intermediate_in_period = any(
                        (w['start'] >= last_processing_start_after_wash and w['start'] <= standalone_due)
                        for w in wash_interruptions
                    )
                    
                    if not has_intermediate_in_period:
                        wash_interruptions.append({
                            'start': standalone_due,
                            'end': standalone_due + intermediate_duration,
                            'type': 'standalone_intermediate'
                        })
                        added_any = True

            if not added_any:
                break

        wash_interruptions.sort(key=lambda x: x['start'])

        total_wash_extension = sum(
            (w['end'] - w['start']).total_seconds() / 3600.0
            for w in wash_interruptions
            if w['start'] >= processing_start
        )

        extended_processing_end = processing_end + timedelta(hours=total_wash_extension)

        for wash in wash_interruptions:
            if wash['type'] == 'scheduled':
                tasks.append({
                    'start': wash['start'],
                    'end': wash['end'],
                    'task': 'wash',
                    'product': 'Scheduled Wash',
                    'order': -2
                })
                tasks.append({
                    'start': wash['start'],
                    'end': wash['end'],
                    'task': 'intermediate_wash',
                    'product': 'Intermediate Wash',
                    'order': -1
                })
                last_wash_end_time = wash['end']
                reset_24h_on_next_processing = True
            elif wash['type'] == 'standalone_intermediate':
                tasks.append({
                    'start': wash['start'],
                    'end': wash['end'],
                    'task': 'intermediate_wash',
                    'product': 'Intermediate Wash',
                    'order': -1
                })
                last_wash_end_time = wash['end']
                reset_24h_on_next_processing = True

        # Create processing segments
        segment_start = processing_start
        for wash in wash_interruptions:
            if segment_start < wash['start']:
                tasks.append({
                    'start': segment_start,
                    'end': wash['start'],
                    'task': 'processing',
                    'product': product_name,
                    'order': i
                })
            segment_start = max(segment_start, wash['end'] + timedelta(minutes=1))

        if segment_start < extended_processing_end:
            tasks.append({
                'start': segment_start,
                'end': extended_processing_end,
                'task': 'processing',
                'product': product_name,
                'order': i
            })

        current_time = extended_processing_end

        if wash_interruptions:
            last_wash = max(wash_interruptions, key=lambda w: w['end'])
            if last_wash['end'] <= segment_start and last_wash.get('type') in ('scheduled', 'standalone_intermediate'):
                last_processing_start_after_wash = segment_start
                reset_24h_on_next_processing = False

        # === Batch markers ===
        batch_size = 30000
        num_batches = max(1, -(-quantity_liters // batch_size))
        if num_batches > 1:
            batch_duration = (extended_processing_end - processing_start) / num_batches
            for b in range(1, num_batches):
                batch_marker_time = processing_start + b * batch_duration
                tasks.append({
                    'start': batch_marker_time,
                    'end': batch_marker_time,
                    'task': 'batch_marker',
                    'product': product_name,
                    'order': i
                })

    # === Visualization ===
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_facecolor('white')

    if not tasks:
        st.error("No tasks generated. Please check your data.")
        return None

    tasks_df = pd.DataFrame(tasks)
    wash_products = ['Scheduled Wash', 'Intermediate Wash']
    other_products = [p for p in tasks_df['product'].unique() if p not in wash_products]
    product_order = [p for p in wash_products if p in tasks_df['product'].unique()] + other_products

    y_pos = 0
    for product_name in product_order:
        product_tasks = tasks_df[tasks_df['product'] == product_name].sort_values('start')

        for _, task in product_tasks.iterrows():
            if task['task'] == 'batch_marker':
                ax.axvline(task['start'], ymin=(y_pos-0.4)/len(product_order), 
                          ymax=(y_pos+0.4)/len(product_order),
                          color='red', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.text(task['start'], y_pos+0.45, "Batch", rotation=90,
                        va='bottom', ha='center', fontsize=7, color='red',
                        fontweight='bold')
                continue

            duration = task['end'] - task['start']
            ax.broken_barh(
                [(task['start'], duration)],
                (y_pos - 0.4, 0.8),
                facecolors=colors[task['task']],
                edgecolor='black'
            )

            if task['task'] in ['wash', 'intermediate_wash']:
                ax.vlines(task['start'], ymin=-0.5, ymax=y_pos + 0.4,
                        color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.vlines(task['end'], ymin=-0.5, ymax=y_pos + 0.4,
                        color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

                ax.text(task['start'], -0.1, task['start'].strftime('%H:%M'),
                        rotation=90, va='top', ha='right', fontsize=8, fontweight='bold',
                        color='black', path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
                ax.text(task['end'], -0.1, task['end'].strftime('%H:%M'),
                        rotation=90, va='top', ha='right', fontsize=8, fontweight='bold',
                        color='black', path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])

            if task['task'] == 'processing':
                ax.text(task['start'], y_pos + 0.5, task['start'].strftime('%H:%M'),
                        rotation=90, va='bottom', ha='left', fontsize=7, fontweight='bold',
                        color='black', path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])
                ax.text(task['end'], y_pos + 0.5, task['end'].strftime('%H:%M'),
                        rotation=90, va='bottom', ha='left', fontsize=7, fontweight='bold',
                        color='black', path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])

        y_pos += 1

    ax.set_yticks(range(len(product_order)))
    ax.set_yticklabels(product_order)
    ax.set_xlabel("Time")
    
    start_date = tasks_df['start'].min()
    end_date = tasks_df['end'].max()
    week_start = start_date.strftime('%Y-%m-%d')
    week_end = end_date.strftime('%Y-%m-%d')
    
    title = f"Weekly Production Plan Timeline\nWeek: {week_start} to {week_end}"
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True)
    
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.text(0.01, 0.02, f"Generated: {current_timestamp}", 
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, alpha=0.7, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

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