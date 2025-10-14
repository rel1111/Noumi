import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import streamlit as st
import io


class ProductionScheduler:
    """Handles production scheduling logic with clean separation of concerns."""
    
    def __init__(self, df):
        self.df = self._clean_dataframe(df)
        self.tasks = []
        self.wash_duration = timedelta(0)
        self.gap_duration = timedelta(0)
        self.intermediate_duration = timedelta(minutes=180)
        self.current_time = None
        self.next_scheduled_wash = None
        self.last_processing_start_after_wash = None
        self.first_wash_time = None
        
        self.colors = {
            'processing': 'darkgreen',
            'wash': 'purple',
            'changeover': 'darkorange',
            'intermediate_wash': 'lightblue'
        }
    
    def _clean_dataframe(self, df):
        """Remove rows with empty/NaN product names."""
        df = df.dropna(subset=['product name'])
        df = df[df['product name'].str.strip() != '']
        df = df.reset_index(drop=True)
        st.info(f"Cleaned data: {len(df)} valid product rows found")
        return df
    
    @staticmethod
    def safe_number_parse(value, default=0):
        """Safely parse numbers that may contain commas."""
        try:
            if pd.isna(value) or value == '':
                return default
            if isinstance(value, str):
                clean_value = value.replace(',', '')
                return int(float(clean_value))
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_datetime_parse(value, field_name="date"):
        """Safely parse datetime values that may be corrupted or in various formats."""
        # Handle empty values, dots, or other placeholder strings
        if pd.isna(value) or value == '' or value == '.' or str(value).strip() in ['.', '-', 'None', 'nan']:
            return None
        
        try:
            # If it's already a valid datetime, validate and return it
            if isinstance(value, (datetime, pd.Timestamp)):
                # Validate it's within reasonable bounds
                dt = pd.to_datetime(value)
                
                # Ignore placeholder dates (Excel autofill creates these)
                if dt.year == 1970 or dt.year < 1900 or dt.year > 2100:
                    return None
                    
                return dt
            
            # Try standard parsing first
            dt = pd.to_datetime(value)
            
            # Ignore placeholder dates
            if dt.year == 1970 or dt.year < 1900 or dt.year > 2100:
                return None
                
            return dt
            
        except (ValueError, pd.errors.OutOfBoundsDatetime, OverflowError, TypeError) as e:
            # If it's a number that might be an Excel serial date
            if isinstance(value, (int, float)):
                try:
                    # Excel dates start from 1899-12-30
                    # Valid Excel dates are between 1 and 2958465 (year 9999)
                    if 1 <= value <= 2958465:
                        dt = pd.Timestamp('1899-12-30') + pd.Timedelta(days=value)
                        if dt.year == 1970 or dt.year < 1900 or dt.year > 2100:
                            return None
                        return dt
                    else:
                        return None
                except:
                    return None
            
            # Try parsing as string
            if isinstance(value, str):
                # Skip placeholder strings
                if value.strip() in ['.', '-', 'None', 'nan', '']:
                    return None
                    
                try:
                    dt = pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S')
                    if dt.year == 1970 or dt.year < 1900:
                        return None
                    return dt
                except:
                    try:
                        dt = pd.to_datetime(value, format='%Y-%m-%d')
                        if dt.year == 1970 or dt.year < 1900:
                            return None
                        return dt
                    except:
                        return None
            
            return None
    
    def parse_schedule_parameters(self):
        """Extract schedule parameters from the first row of data ONLY."""
        try:
            # Parse start date from first row only
            self.current_time = self.safe_datetime_parse(self.df.loc[0, 'Date from'], 'Date from')
            if self.current_time is None:
                st.error("Could not parse 'Date from' field in first row. Please check the date format.")
                return False
            
            # Parse wash parameters from first row only
            wash_duration_mins = self.safe_number_parse(self.df.loc[0, 'Duration'], 0)
            wash_gap_mins = self.safe_number_parse(self.df.loc[0, 'Gap'], 0)
            
            self.wash_duration = timedelta(minutes=wash_duration_mins)
            self.gap_duration = timedelta(minutes=wash_gap_mins)
            
            # Parse first wash time from first row only (if specified)
            self.first_wash_time = None
            if 'First Wash Time' in self.df.columns:
                first_wash_raw = self.df.loc[0, 'First Wash Time']
                # Only parse if it's not a placeholder
                if pd.notna(first_wash_raw) and str(first_wash_raw).strip() not in ['.', '-', '', 'None', 'nan']:
                    self.first_wash_time = self.safe_datetime_parse(first_wash_raw, 'First Wash Time')
                    if self.first_wash_time:
                        st.info(f"First wash scheduled at: {self.first_wash_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self.first_wash_time is None:
                st.info("No first wash time specified - production will start immediately")
            
            # Set up scheduled wash tracking
            if self.first_wash_time and self.wash_duration > timedelta(0):
                self.next_scheduled_wash = self.first_wash_time + self.wash_duration + self.gap_duration
            elif self.wash_duration > timedelta(0):
                self.next_scheduled_wash = self.current_time + self.gap_duration
            else:
                self.next_scheduled_wash = None
            
            st.info(f"Production starts at: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(f"Wash duration: {wash_duration_mins} minutes, Gap: {wash_gap_mins} minutes")
            st.info(f"Intermediate wash duration: 180 minutes (3 hours)")
            
            return True
            
        except Exception as e:
            st.error(f"Error parsing schedule parameters from first row: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def add_full_wash(self, start_time, wash_type='scheduled'):
        """
        Add a complete wash (both scheduled wash and intermediate wash layers).
        
        Args:
            start_time: When the wash starts
            wash_type: Type identifier for tracking ('scheduled', 'first_wash', 'additional')
        
        Returns:
            The end time of the wash
        """
        if self.wash_duration <= timedelta(0):
            return start_time
        
        end_time = start_time + self.wash_duration
        
        # Add scheduled wash layer
        self.tasks.append({
            'start': start_time,
            'end': end_time,
            'task': 'wash',
            'product': 'Scheduled Wash',
            'order': -2,
            'wash_type': wash_type
        })
        
        # Add intermediate wash layer (visual overlay)
        self.tasks.append({
            'start': start_time,
            'end': end_time,
            'task': 'intermediate_wash',
            'product': 'Intermediate Wash',
            'order': -1,
            'wash_type': wash_type
        })
        
        return end_time
    
    def add_intermediate_wash(self, start_time):
        """
        Add a standalone 24-hour intermediate wash.
        
        Args:
            start_time: When the intermediate wash starts
        
        Returns:
            The end time of the wash
        """
        end_time = start_time + self.intermediate_duration
        
        self.tasks.append({
            'start': start_time,
            'end': end_time,
            'task': 'intermediate_wash',
            'product': 'Intermediate Wash',
            'order': -1,
            'wash_type': 'standalone_intermediate'
        })
        
        return end_time
    
    def add_changeover(self, start_time, duration_mins, product_name, order):
        """
        Add a changeover task.
        
        Args:
            start_time: When changeover starts
            duration_mins: Duration in minutes
            product_name: Product being changed to
            order: Product order index
        
        Returns:
            The end time of the changeover
        """
        if duration_mins <= 0:
            return start_time
        
        end_time = start_time + timedelta(minutes=duration_mins)
        
        self.tasks.append({
            'start': start_time,
            'end': end_time,
            'task': 'changeover',
            'product': product_name,
            'order': order
        })
        
        return end_time
    
    def add_processing_segment(self, start_time, end_time, product_name, order):
        """
        Add a processing segment.
        
        Args:
            start_time: When processing starts
            end_time: When processing ends
            product_name: Product being processed
            order: Product order index
        """
        if start_time >= end_time:
            return
        
        self.tasks.append({
            'start': start_time,
            'end': end_time,
            'task': 'processing',
            'product': product_name,
            'order': order
        })
    
    def schedule_first_wash(self):
        """Schedule the first wash if specified, without halting production."""
        if self.first_wash_time and self.wash_duration > timedelta(0):
            self.add_full_wash(self.first_wash_time, wash_type='first_wash')
    
    def get_pending_scheduled_washes(self, until_time):
        """
        Get all scheduled washes that should occur before a given time.
        
        Args:
            until_time: Time to check up to
        
        Returns:
            List of wash start times
        """
        washes = []
        while self.next_scheduled_wash and self.next_scheduled_wash <= until_time:
            washes.append(self.next_scheduled_wash)
            wash_end = self.next_scheduled_wash + self.wash_duration
            self.next_scheduled_wash = wash_end + self.gap_duration
        
        return washes
    
    def calculate_wash_interruptions(self, processing_start, processing_end, product_index):
        """
        Calculate all wash interruptions for a processing window.
        
        Args:
            processing_start: When processing would start
            processing_end: When processing would end (without interruptions)
            product_index: Index of the product
        
        Returns:
            List of wash interruption dictionaries with 'start', 'end', 'type'
        """
        interruptions = []
        
        # Include first wash if it interrupts this product
        if self.first_wash_time and self.wash_duration > timedelta(0):
            first_wash_end = self.first_wash_time + self.wash_duration
            if processing_start < first_wash_end and self.first_wash_time < processing_end:
                interruptions.append({
                    'start': self.first_wash_time,
                    'end': first_wash_end,
                    'type': 'first_wash'
                })
        
        # Determine 24hr timer origin
        temp_24h_origin = self.last_processing_start_after_wash
        if interruptions and processing_start < interruptions[0]['end']:
            # First wash interrupts, so 24hr timer starts after it
            temp_24h_origin = interruptions[0]['end'] + timedelta(minutes=1)
        
        # Iteratively add washes that fit in the extended window
        while True:
            added_any = False
            
            # Calculate current window end including all wash extensions
            total_wash_hours = sum(
                (w['end'] - w['start']).total_seconds() / 3600.0 
                for w in interruptions
            )
            window_end = processing_end + timedelta(hours=total_wash_hours)
            
            # Add scheduled washes within the window
            temp_next = self.next_scheduled_wash
            while temp_next and temp_next < window_end:
                wash_end = temp_next + self.wash_duration
                interruptions.append({
                    'start': temp_next,
                    'end': wash_end,
                    'type': 'scheduled'
                })
                temp_next = wash_end + self.gap_duration
                added_any = True
            
            # Recalculate window end
            total_wash_hours = sum(
                (w['end'] - w['start']).total_seconds() / 3600.0 
                for w in interruptions
            )
            window_end = processing_end + timedelta(hours=total_wash_hours)
            
            # Check for 24hr intermediate wash
            if temp_24h_origin:
                standalone_due = temp_24h_origin + timedelta(hours=24)
                
                if processing_start <= standalone_due <= window_end:
                    # Check if there's already a wash covering this time
                    has_wash_at_due = any(
                        w['start'] >= temp_24h_origin and w['start'] <= standalone_due
                        for w in interruptions
                    )
                    
                    if not has_wash_at_due:
                        interruptions.append({
                            'start': standalone_due,
                            'end': standalone_due + self.intermediate_duration,
                            'type': 'standalone_intermediate'
                        })
                        added_any = True
            
            if not added_any:
                break
        
        # Sort by start time
        interruptions.sort(key=lambda x: x['start'])
        return interruptions
    
    def process_product(self, row, product_index):
        """
        Process a single product through the schedule.
        
        Args:
            row: DataFrame row containing product data
            product_index: Index of the product
        """
        product_name = row['product name']
        quantity_liters = self.safe_number_parse(row['quantity liters'], 0)
        process_speed = self.safe_number_parse(row['process speed per hour'], 1)
        line_efficiency = float(str(row['line efficiency']).replace(',', '')) if pd.notna(row['line efficiency']) else 0.0
        change_over_mins = self.safe_number_parse(row['Change Over'], 0)
        additional_wash = row.get('Additional Wash', 'No') if 'Additional Wash' in row else 'No'
        
        # Insert any pending scheduled washes
        pending_washes = self.get_pending_scheduled_washes(self.current_time)
        for wash_start in pending_washes:
            wash_end = self.add_full_wash(wash_start, wash_type='scheduled')
            self.current_time = max(self.current_time, wash_end + timedelta(minutes=1))
            self.last_processing_start_after_wash = self.current_time
        
        # Add additional wash if requested
        if additional_wash == 'Yes' and self.wash_duration > timedelta(0):
            wash_end = self.add_full_wash(self.current_time, wash_type='additional')
            self.current_time = wash_end + timedelta(minutes=1)
            self.last_processing_start_after_wash = self.current_time
        
        # Add changeover (skip for first product)
        if product_index > 0:
            changeover_end = self.add_changeover(
                self.current_time, 
                change_over_mins, 
                product_name, 
                product_index
            )
            self.current_time = changeover_end
        
        # Calculate processing time
        effective_speed = process_speed * line_efficiency
        if effective_speed == 0:
            st.error(f"Effective speed for product '{product_name}' is zero.")
            return False
        
        processing_hours = quantity_liters / effective_speed
        processing_start = self.current_time
        processing_end = self.current_time + timedelta(hours=processing_hours)
        
        # Initialize 24hr timer if needed
        if self.last_processing_start_after_wash is None:
            self.last_processing_start_after_wash = processing_start
        
        # Calculate all wash interruptions
        interruptions = self.calculate_wash_interruptions(
            processing_start, 
            processing_end, 
            product_index
        )
        
        # Add wash tasks to the schedule
        for wash in interruptions:
            if wash['type'] in ['scheduled', 'first_wash']:
                self.add_full_wash(wash['start'], wash_type=wash['type'])
            elif wash['type'] == 'standalone_intermediate':
                self.add_intermediate_wash(wash['start'])
        
        # Update next_scheduled_wash pointer based on added washes
        scheduled_washes = [w for w in interruptions if w['type'] == 'scheduled']
        if scheduled_washes:
            last_scheduled = max(scheduled_washes, key=lambda w: w['end'])
            self.next_scheduled_wash = last_scheduled['end'] + self.gap_duration
        
        # Calculate total wash extension
        total_wash_extension = sum(
            (w['end'] - w['start']).total_seconds() / 3600.0
            for w in interruptions
            if w['start'] >= processing_start
        )
        extended_processing_end = processing_end + timedelta(hours=total_wash_extension)
        
        # Create processing segments around washes
        segment_start = processing_start
        for wash in interruptions:
            if segment_start < wash['start']:
                self.add_processing_segment(
                    segment_start, 
                    wash['start'], 
                    product_name, 
                    product_index
                )
            segment_start = max(segment_start, wash['end'] + timedelta(minutes=1))
        
        # Final processing segment
        if segment_start < extended_processing_end:
            self.add_processing_segment(
                segment_start, 
                extended_processing_end, 
                product_name, 
                product_index
            )
        
        # Update current time and 24hr origin
        self.current_time = extended_processing_end
        
        if interruptions:
            last_wash = max(interruptions, key=lambda w: w['end'])
            if last_wash['end'] <= segment_start:
                self.last_processing_start_after_wash = segment_start
        
        return True
    
    def generate_schedule(self):
        """Generate the complete production schedule."""
        if not self.parse_schedule_parameters():
            return None
        
        # Schedule first wash (doesn't halt production)
        self.schedule_first_wash()
        
        # Process each product
        for i, row in self.df.iterrows():
            if not self.process_product(row, i):
                return None
        
        return self.tasks


class TimelineVisualizer:
    """Handles visualization of the production timeline."""
    
    def __init__(self, tasks, df, colors):
        self.tasks = tasks
        self.df = df
        self.colors = colors
    
    @staticmethod
    def safe_number_parse(value, default=0):
        """Safely parse numbers that may contain commas."""
        try:
            if pd.isna(value) or value == '':
                return default
            if isinstance(value, str):
                clean_value = value.replace(',', '')
                return int(float(clean_value))
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def create_timeline(self):
        """Create the matplotlib timeline visualization."""
        if not self.tasks:
            st.error("No tasks generated. Please check your data.")
            return None
        
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_facecolor('white')
        
        tasks_df = pd.DataFrame(self.tasks)
        
        # Order products: washes first, then others
        wash_products = ['Scheduled Wash', 'Intermediate Wash']
        other_products = [p for p in tasks_df['product'].unique() if p not in wash_products]
        product_order = [p for p in wash_products if p in tasks_df['product'].unique()] + other_products
        
        # Draw each product row
        y_pos = 0
        for product_name in product_order:
            product_tasks = tasks_df[tasks_df['product'] == product_name].sort_values('start')
            
            # Calculate batches for non-wash products
            num_batches = 0
            if product_name not in wash_products:
                product_row = self.df[self.df['product name'] == product_name]
                if not product_row.empty:
                    quantity_liters = self.safe_number_parse(product_row.iloc[0]['quantity liters'], 0)
                    num_batches = int(quantity_liters / 30000) + (1 if quantity_liters % 30000 > 0 else 0)
            
            self._draw_product_row(ax, product_tasks, y_pos, product_name, num_batches)
            y_pos += 1
        
        # Format axes
        self._format_axes(ax, tasks_df, product_order)
        
        plt.tight_layout()
        return fig
    
    def _draw_product_row(self, ax, product_tasks, y_pos, product_name, num_batches):
        """Draw a single product row with all its tasks."""
        for _, task in product_tasks.iterrows():
            duration = task['end'] - task['start']
            
            # Draw task bar
            ax.broken_barh(
                [(task['start'], duration)],
                (y_pos - 0.4, 0.8),
                facecolors=self.colors[task['task']],
                edgecolor='black'
            )
            
            # Add batch indicators for processing
            if task['task'] == 'processing' and num_batches > 0:
                self._draw_batch_indicators(ax, product_tasks, task, y_pos, num_batches)
            
            # Add time labels
            self._draw_time_labels(ax, task, y_pos)
    
    def _draw_batch_indicators(self, ax, product_tasks, current_task, y_pos, num_batches):
        """Draw batch boundary indicators on processing bars."""
        processing_segments = product_tasks[product_tasks['task'] == 'processing'].sort_values('start')
        
        # Calculate total processing time
        total_processing_duration = sum(
            (seg['end'] - seg['start']).total_seconds() 
            for _, seg in processing_segments.iterrows()
        )
        
        time_per_batch = total_processing_duration / num_batches
        
        # Calculate cumulative time before this segment
        cumulative_time = 0
        for _, seg in processing_segments.iterrows():
            if seg['start'] == current_task['start']:
                break
            cumulative_time += (seg['end'] - seg['start']).total_seconds()
        
        # Draw batch markers within this segment
        segment_duration = (current_task['end'] - current_task['start']).total_seconds()
        batch_start_time = cumulative_time
        
        for batch_num in range(1, num_batches + 1):
            batch_end_time = batch_num * time_per_batch
            
            # Check if batch boundary is in this segment
            if batch_start_time < batch_end_time <= batch_start_time + segment_duration:
                time_into_segment = batch_end_time - batch_start_time
                batch_time = current_task['start'] + timedelta(seconds=time_into_segment)
                
                # Draw batch line
                ax.vlines(batch_time, ymin=y_pos - 0.4, ymax=y_pos + 0.4,
                         color='white', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add batch label
                ax.text(batch_time, y_pos, f'B{batch_num}',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='darkgreen',
                               edgecolor='white', linewidth=1.5, alpha=0.9))
    
    def _draw_time_labels(self, ax, task, y_pos):
        """Draw time labels for tasks."""
        if task['task'] in ['wash', 'intermediate_wash']:
            # Vertical lines and labels for washes
            ax.vlines(task['start'], ymin=-0.5, ymax=y_pos + 0.4,
                     color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.vlines(task['end'], ymin=-0.5, ymax=y_pos + 0.4,
                     color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            ax.text(task['start'], -0.1, task['start'].strftime('%H:%M'),
                   rotation=90, va='top', ha='right', fontsize=8, fontweight='bold',
                   color='black', 
                   path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
            ax.text(task['end'], -0.1, task['end'].strftime('%H:%M'),
                   rotation=90, va='top', ha='right', fontsize=8, fontweight='bold',
                   color='black',
                   path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='yellow')])
        
        elif task['task'] == 'processing':
            # Labels for processing
            ax.text(task['start'], y_pos + 0.5, task['start'].strftime('%H:%M'),
                   rotation=90, va='bottom', ha='left', fontsize=7, fontweight='bold',
                   color='black',
                   path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])
            ax.text(task['end'], y_pos + 0.5, task['end'].strftime('%H:%M'),
                   rotation=90, va='bottom', ha='left', fontsize=7, fontweight='bold',
                   color='black',
                   path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground='lightgreen')])
    
    def _format_axes(self, ax, tasks_df, product_order):
        """Format the axes, grid, and labels."""
        # Y-axis
        ax.set_yticks(range(len(product_order)))
        ax.set_yticklabels(product_order)
        ax.invert_yaxis()
        
        # Title
        start_date = tasks_df['start'].min()
        end_date = tasks_df['end'].max()
        week_start = start_date.strftime('%Y-%m-%d')
        week_end = end_date.strftime('%Y-%m-%d')
        
        title = f"Weekly Production Plan Timeline\nWeek: {week_start} to {week_end}"
        ax.set_title(title, fontsize=14, pad=20)
        
        # Grid
        ax.grid(True)
        
        # Timestamp
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.01, 0.02, f"Generated: {current_timestamp}",
               transform=ax.transAxes, ha='left', va='bottom',
               fontsize=9, alpha=0.7, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # X-axis time formatting
        ax.set_xlabel("Time")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        
        # Day dividers
        first_date = tasks_df['start'].min().floor('D')
        last_date = tasks_df['end'].max().ceil('D')
        delta_days = (last_date - first_date).days + 1
        for day in range(delta_days):
            day_start = first_date + timedelta(days=day)
            ax.axvline(day_start, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        
        plt.xticks(rotation=90, ha='right', va='top')
        plt.setp(ax.get_xminorticklabels(), rotation=90, ha='right', va='top')
        
        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, fc=self.colors[t]) for t in self.colors]
        ax.legend(handles, self.colors.keys(), loc='upper right')


def generate_timeline(df):
    """
    Main function to generate production timeline.
    
    Args:
        df: DataFrame containing production plan data
    
    Returns:
        matplotlib figure or None if error
    """
    # Create scheduler and generate schedule
    scheduler = ProductionScheduler(df)
    tasks = scheduler.generate_schedule()
    
    if tasks is None:
        return None
    
    # Create visualization
    visualizer = TimelineVisualizer(tasks, df, scheduler.colors)
    return visualizer.create_timeline()


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
            # Read file with better error handling
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Read Excel without automatic date parsing to avoid corruption
                df = pd.read_excel(uploaded_file, date_parser=None)
            
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Show raw data for debugging
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Show data types for debugging
            with st.expander("Column Data Types (for debugging)"):
                st.write(df.dtypes)
                st.write("\n**First row values:**")
                for col in df.columns:
                    st.write(f"- **{col}**: {df.loc[0, col]} (type: {type(df.loc[0, col]).__name__})")
            
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
            st.write("**Debug info:**")
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error details: {str(e)}")
            st.write("\nThis often happens when date fields are corrupted. Try:")
            st.write("1. Check that 'Date from' and 'First Wash Time' contain valid dates")
            st.write("2. Save your Excel file as a new copy")
            st.write("3. Export to CSV and upload the CSV instead")
    else:
        st.info("Please upload a file to get started.")
        
        with st.expander("Expected File Format"):
            st.write("Your file should contain these columns:")
            st.write("**Required:** product name, quantity liters, process speed per hour, line efficiency, Change Over, Date from, Duration, Gap")
            st.write("**Optional:** First Wash Time, Additional Wash")
            st.write("**Data Format:** All wash parameters (Duration, Gap, First Wash Time) should be in the first row.")
            st.write("\n**Date Format:** Dates should be in format YYYY-MM-DD HH:MM:SS or YYYY-MM-DD")


if __name__ == "__main__":
    main()