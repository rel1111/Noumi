# streamlit_production_timeline.py
# Production Timeline Generator Streamlit App
# Features implemented:
# - Upload Excel with product rows (see user's example columns)
# - Parse product data and wash system parameters (flexible to missing fields)
# - Simulate processing with interruptions from scheduled/additional/intermediate washes
# - Apply changeovers and overlap rules (skip changeover if a wash occurs during the changeover window)
# - Produce an interactive Gantt timeline (Plotly) and downloadable event table

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Production Timeline Generator", layout="wide")
st.title("Production Timeline Generator")
st.write("Upload an Excel file with product rows following the example in your spec. The app will simulate processing, washes, and changeovers and show a Gantt chart.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xls","xlsx","csv"]) 

# Helper functions

def to_datetime_maybe(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    try:
        return pd.to_datetime(str(x))
    except Exception:
        return None


def minutes(td):
    return td.total_seconds() / 60.0


def schedule_recurring_washes(first_wash_time, wash_duration_min, wash_gap_min, until_time):
    """Return list of (start, end) for recurring washes from first_wash_time until until_time.
       Interval = wash_duration + wash_gap (they described gap as time between washes)
    """
    events = []
    if first_wash_time is None:
        return events
    interval = wash_duration_min + wash_gap_min
    cur = first_wash_time
    while cur <= until_time:
        events.append((cur, cur + timedelta(minutes=wash_duration_min)))
        cur = cur + timedelta(minutes=interval)
    return events


def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals, key=lambda x: x[0])
    out = [s[0]]
    for a,b in s[1:]:
        last_s, last_e = out[-1]
        if a <= last_e:
            if b > last_e:
                out[-1] = (last_s, b)
        else:
            out.append((a,b))
    return out


def find_next_wash_after(timepoint, washes):
    for s,e in washes:
        if e > timepoint and s >= timepoint:
            return (s,e)
    return None


def washes_during_interval(start, end, washes):
    """Return list of washes that overlap [start, end)"""
    out = []
    for s,e in washes:
        if s < end and e > start:
            out.append((s,e))
    return out


def simulate_product_processing(start_time, processing_minutes, global_washes, additional_wash_flag=False, wash_duration_min=0, intermediate_min=180):
    """
    Simulate processing for a single product starting at start_time for processing_minutes of active processing.
    global_washes: list of (start,end) of scheduled washes across timeline
    additional_wash_flag: if True schedule a one-time additional wash at start_time (duration wash_duration_min)
    intermediate_min: duration of intermediate wash (always simultaneous with other washes)

    Returns: segments list of dicts with keys: type ('processing' or 'wash'), start, end
    and list of wash events that apply to this product (start,end)
    """
    segments = []
    wash_events = []
    remaining = processing_minutes
    cur = start_time

    # If additional wash at start -> schedule it immediately
    if additional_wash_flag:
        # combined duration is max(wash_duration_min, intermediate_min)
        combined = max(wash_duration_min, intermediate_min)
        wash_start = cur
        wash_end = cur + timedelta(minutes=combined)
        wash_events.append((wash_start, wash_end))
        segments.append({"type":"wash", "start":wash_start, "end":wash_end})
        cur = wash_end

    # Track last time a wash happened for the standalone-intermediate rule
    last_wash_time = cur if wash_events else None
    time_since_last_wash = 0.0

    # We'll iterate: at each step find the next scheduled wash that starts >= cur and see if it happens while processing
    # If a scheduled wash starts during processing, we process until the wash start, then add wash extension and continue
    # Also handle standalone intermediate wash: if 24 hours (1440 minutes) of processing with 0 washes -> trigger intermediate

    # Build a list of scheduled washes that can affect this product (those whose start < some reasonable horizon)
    # We'll loop until remaining <= 0
    horizon = cur + timedelta(days=14)  # safety cap

    # Precompute scheduled washes that might affect product
    scheduled_affecting = [w for w in global_washes if w[1] >= cur]
    scheduled_affecting = sorted(scheduled_affecting, key=lambda x: x[0])
    idx = 0

    # If last_wash_time is None set to start_time
    last_wash_time = cur if last_wash_time is None else last_wash_time
    processed_since_last_wash = 0.0

    while remaining > 1e-6:
        # Check standalone intermediate: if processed_since_last_wash >= 1440 -> schedule immediate intermediate wash
        if processed_since_last_wash >= 1440:
            # schedule intermediate now
            wash_start = cur
            wash_end = cur + timedelta(minutes=intermediate_min)
            wash_events.append((wash_start, wash_end))
            segments.append({"type":"wash", "start":wash_start, "end":wash_end})
            cur = wash_end
            processed_since_last_wash = 0.0
            last_wash_time = cur
            continue

        # Find next scheduled wash that starts after or at cur
        next_wash = None
        while idx < len(scheduled_affecting) and scheduled_affecting[idx][1] <= cur:
            idx += 1
        if idx < len(scheduled_affecting):
            next_wash = scheduled_affecting[idx]

        if not next_wash:
            # No more scheduled washes -> finish processing
            seg_start = cur
            seg_end = cur + timedelta(minutes=remaining)
            segments.append({"type":"processing","start":seg_start,"end":seg_end})
            cur = seg_end
            remaining = 0
            processed_since_last_wash += minutes(seg_end - seg_start)
            break
        else:
            wash_start, wash_end = next_wash
            # If wash starts after finishing processing -> finish
            time_until_wash = minutes(wash_start - cur)
            if time_until_wash >= remaining:
                seg_start = cur
                seg_end = cur + timedelta(minutes=remaining)
                segments.append({"type":"processing","start":seg_start,"end":seg_end})
                cur = seg_end
                processed_since_last_wash += remaining
                remaining = 0
                break
            else:
                # process up to wash_start
                if time_until_wash > 0:
                    seg_start = cur
                    seg_end = wash_start
                    segments.append({"type":"processing","start":seg_start,"end":seg_end})
                    remaining -= time_until_wash
                    processed_since_last_wash += time_until_wash
                    cur = wash_start
                # At wash_start, scheduled wash occurs; it always includes simultaneous intermediate (take combined duration)
                combined = max(minutes(wash_end - wash_start), intermediate_min)
                wash_real_end = wash_start + timedelta(minutes=combined)
                wash_events.append((wash_start, wash_real_end))
                segments.append({"type":"wash","start":wash_start,"end":wash_real_end})
                cur = wash_real_end
                processed_since_last_wash = 0.0
                idx += 1
                # After wash, continue loop
    return segments, wash_events


if uploaded_file is not None:
    # Load file - support csv or excel
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(20))

    # Normalize column names (lowercase, strip)
    df.columns = [str(c).strip() for c in df.columns]

    # Required columns mapping heuristics
    colmap = {}
    lowcols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in lowcols:
                return lowcols[n.lower()]
        return None

    colmap['product'] = pick('product name','product','name')
    colmap['quantity'] = pick('quantity liters','quantity','liters','quantity_liters')
    colmap['speed'] = pick('process speed per hour','process speed per hour','process speed','speed')
    colmap['eff'] = pick('line efficiency','line efficiency (%)','efficiency','line efficiency (%)')
    colmap['changeover'] = pick('Change Over','changeover','change over','change_over')
    colmap['datefrom'] = pick('Date from','date from','start','start_time')
    colmap['duration'] = pick('Duration','duration')
    colmap['gap'] = pick('Gap','gap')
    colmap['firstwash'] = pick('First Wash Time','First Wash Time','first wash time','first_wash_time')
    colmap['additional'] = pick('Additional Wash','Additional Wash','additional wash','additional')

    st.write("Detected columns:")
    st.json(colmap)

    # Fill defaults and convert
    products = []
    # Extract wash system params from first non-empty row
    wash_duration_min = None
    wash_gap_min = None
    first_wash_time = None

    for i,row in df.iterrows():
        name = row.get(colmap['product']) if colmap['product'] else f'Product {i}'
        qty = row.get(colmap['quantity']) if colmap['quantity'] else 0
        try:
            qty = float(qty)
        except:
            qty = 0.0
        speed = row.get(colmap['speed']) if colmap['speed'] else 1.0
        try:
            speed = float(speed)
        except:
            speed = 1.0
        eff = row.get(colmap['eff']) if colmap['eff'] else 1.0
        try:
            eff = float(eff)
        except:
            eff = 1.0
        # If eff seems in 0-1 but user might type 70 for 70% -> normalize
        if eff > 1.0:
            eff = eff / 100.0
        changeover_min = row.get(colmap['changeover']) if colmap['changeover'] else 0
        try:
            changeover_min = float(changeover_min)
        except:
            changeover_min = 0.0
        datefrom = to_datetime_maybe(row.get(colmap['datefrom'])) if colmap['datefrom'] else None
        duration_val = row.get(colmap['duration']) if colmap['duration'] else None
        gap_val = row.get(colmap['gap']) if colmap['gap'] else None
        firstwash_val = to_datetime_maybe(row.get(colmap['firstwash'])) if colmap['firstwash'] else None
        additional_flag = str(row.get(colmap['additional'])).strip().lower() in ['yes','y','true','1'] if colmap['additional'] else False

        # opportunistically set global wash params from any row that has them (first seen)
        if wash_duration_min is None and duration_val not in [None,0,0.0,'',np.nan]:
            try:
                wash_duration_min = float(duration_val)
            except:
                pass
        if wash_gap_min is None and gap_val not in [None,0,0.0,'',np.nan]:
            try:
                wash_gap_min = float(gap_val)
            except:
                pass
        if first_wash_time is None and firstwash_val is not None:
            first_wash_time = firstwash_val

        products.append({
            'name': str(name),
            'qty': qty,
            'speed': speed,
            'eff': eff,
            'changeover_min': changeover_min,
            'datefrom': datefrom,
            'additional': additional_flag
        })

    # Set defaults if null
    if wash_duration_min is None:
        wash_duration_min = st.number_input('Scheduled wash duration (minutes)', value=360)
    else:
        st.write(f"Detected wash duration (minutes): {wash_duration_min}")
    if wash_gap_min is None:
        wash_gap_min = st.number_input('Scheduled wash gap (minutes)', value=3120)
    else:
        st.write(f"Detected wash gap (minutes): {wash_gap_min}")
    if first_wash_time is None:
        first_wash_time = st.datetime_input('First scheduled wash time (optional)', value=None)
        if first_wash_time == pd.NaT:
            first_wash_time = None
    else:
        st.write(f"Detected first wash time: {first_wash_time}")

    intermediate_min = 180
    st.write(f"Intermediate wash duration is fixed at {intermediate_min} minutes (3 hours) and always runs simultaneously with scheduled/additional washes.")

    # Decide timeline start. If first product has datefrom use that, else ask user for start
    timeline_start = None
    if products and products[0]['datefrom'] is not None:
        timeline_start = products[0]['datefrom']
    else:
        timeline_start = st.datetime_input('Timeline start (used if first product has no Date from)', value=datetime.now())

    # Build recurring wash list across full horizon. We need an approximate end horizon: estimate naive processing length sum
    total_processing_minutes_est = 0
    for p in products:
        proc_min = (p['qty'] / (p['speed'] * p['eff'])) * 60.0
        total_processing_minutes_est += proc_min + p['changeover_min']
    # add some buffer days
    horizon_end = timeline_start + timedelta(minutes=total_processing_minutes_est * 1.3 + 1440)

    recurring_washes = schedule_recurring_washes(first_wash_time, wash_duration_min, wash_gap_min, horizon_end) if first_wash_time else []

    # We'll simulate sequentially product by product
    all_segments = []
    all_washes_global = list(recurring_washes)  # we'll append additional/standalone washes as we find them

    prev_end = None
    for idx,p in enumerate(products):
        # Determine start
        if idx == 0:
            if p['datefrom'] is not None:
                start = p['datefrom']
            else:
                start = timeline_start
        else:
            # default candidate start = prev_end + this product's changeover
            candidate = prev_end + timedelta(minutes=p['changeover_min'])
            # If any recurring wash overlaps changeover window (prev_end, candidate), skip changeover and use wash_end instead.
            overlapping_washes = washes_during_interval(prev_end, candidate, all_washes_global)
            if overlapping_washes:
                # use the end of the last overlapping wash as the start
                wash_end = max([w[1] for w in overlapping_washes])
                start = wash_end
            else:
                # also if the product row itself has a datefrom provided and it's later than candidate use that
                if p['datefrom'] is not None and p['datefrom'] > candidate:
                    start = p['datefrom']
                else:
                    start = candidate

        # compute raw processing minutes
        processing_minutes = (p['qty'] / (p['speed'] * p['eff'])) * 60.0
        # simulate product
        segs, wash_events = simulate_product_processing(start, processing_minutes, all_washes_global, additional_wash_flag=p['additional'], wash_duration_min=wash_duration_min, intermediate_min=intermediate_min)

        # register produced segments to global list and add wash events to all_washes_global
        for s in segs:
            s_rec = s.copy()
            s_rec['product'] = p['name']
            all_segments.append(s_rec)
        for w in wash_events:
            all_washes_global.append(w)
        # Keep merged global washes sorted and merged to avoid duplicates/overlaps
        all_washes_global = merge_intervals(all_washes_global)

        # set prev_end as the end of last segment for this product
        last_seg = segs[-1]
        prev_end = last_seg['end']

    # Build dataframe for visualization
    rows = []
    # Add a top-level 'Washes' track entries (all_washes_global) for the Gantt
    for s,e in all_washes_global:
        rows.append({"Task":"WASH", "Start":s, "Finish":e, "Type":"wash", "Product":"WASH"})
    # Add product segments
    for s in all_segments:
        typ = s['type']
        rows.append({"Task": s.get('product', 'Product'), "Start": s['start'], "Finish": s['end'], "Type": typ, "Product": s.get('product')})

    viz_df = pd.DataFrame(rows)
    # Sort by start time so Plotly draws in order
    viz_df = viz_df.sort_values('Start')

    st.subheader("Event table (for debugging)")
    st.dataframe(viz_df)

    st.subheader("Gantt Timeline")
    # For nicer ordering, we will create an ordered category for Task where WASH comes first, then each product in the order they were input
    product_order = ["WASH"] + [p['name'] for p in products]
    viz_df['Task'] = pd.Categorical(viz_df['Task'], categories=product_order, ordered=True)

    color_map = {'processing':'green','wash':'purple'}
    # Map unknown types to processing
    viz_df['color_type'] = viz_df['Type'].map(lambda x: x if x in color_map else 'processing')

    fig = px.timeline(viz_df, x_start="Start", x_end="Finish", y="Task", color="color_type", hover_data=["Product","Type","Start","Finish"] )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # Provide CSV download of events
    csv = viz_df.to_csv(index=False)
    st.download_button("Download events CSV", data=csv, file_name="timeline_events.csv", mime="text/csv")

    st.success("Simulation complete. Inspect the Gantt chart and event table. Adjust wash duration/gap/first wash inputs if needed and re-upload or change inputs.")

else:
    st.info("Upload an example Excel or CSV (see the sample format in your message). The app will attempt to auto-detect columns.")
