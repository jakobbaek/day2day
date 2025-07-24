#!/usr/bin/env python3
"""
Debug script to trace NOVO data through the processing pipeline.
This helps identify where data gets systematically cut off at 13:00.
"""

from day2day.data.datetime_utils import DateTimeStandardizer
from day2day.data.preparation import DataPreparator
import polars as pl
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def main():
    print("🔍 DEBUGGING NOVO DATA PIPELINE")
    print("=" * 50)
    
    # Initialize components
    preparator = DataPreparator()
    standardizer = DateTimeStandardizer()

    # === STEP 1: Check raw data ===
    print("\n=== STEP 1: Loading raw data ===")
    try:
        raw_df = preparator.load_raw_data('danish_stocks_1m.csv', standardize_datetime=False)
        print(f"✓ Raw data loaded: {len(raw_df)} rows")
        
        # Check NOVO in raw data
        novo_raw = raw_df.filter(pl.col('ticker') == 'NOVOb:xcse')
        if len(novo_raw) > 0:
            print(f"✓ Found NOVO raw data: {len(novo_raw)} records")
            
            # Check if datetime is already parsed or string
            datetime_dtype = novo_raw.select(pl.col('datetime')).dtypes[0]
            print(f"📊 Datetime column type: {datetime_dtype}")
            
            if datetime_dtype == pl.Datetime:
                # Already parsed datetime
                recent_dates = novo_raw.select(
                    pl.col('datetime').dt.date().unique().alias('date')
                ).sort('date').tail(3).to_series().to_list()
                
                target_date = recent_dates[-1]
                daily_raw = novo_raw.filter(pl.col('datetime').dt.date() == target_date)
            else:
                # String datetime
                recent_dates = novo_raw.select(
                    pl.col('datetime').str.slice(0, 10).unique().alias('date')
                ).sort('date').tail(3).to_series().to_list()
                
                target_date = recent_dates[-1]
                daily_raw = novo_raw.filter(pl.col('datetime').str.contains(str(target_date)))
            
            print(f"📅 Recent dates with NOVO data: {recent_dates}")
            print(f"\n🔍 Analyzing NOVO data on {target_date}:")
            print(f"Raw records on {target_date}: {len(daily_raw)}")
            
            if len(daily_raw) > 0:
                # Show hourly distribution in raw data
                if datetime_dtype == pl.Datetime:
                    # Use datetime methods
                    daily_raw = daily_raw.with_columns([
                        pl.col('datetime').dt.hour().alias('hour'),
                        pl.col('datetime').dt.minute().alias('minute')
                    ])
                else:
                    # Use string slicing
                    daily_raw = daily_raw.with_columns([
                        pl.col('datetime').str.slice(11, 2).cast(pl.Int32).alias('hour'),
                        pl.col('datetime').str.slice(14, 2).cast(pl.Int32).alias('minute')
                    ])
                
                hourly_raw = daily_raw.group_by('hour').agg(pl.len().alias('count')).sort('hour')
                print("\n📊 Raw data hourly distribution:")
                for row in hourly_raw.iter_rows(named=True):
                    print(f"  Hour {row['hour']:02d}: {row['count']} records")
                
                # Show time range
                first_time = daily_raw.select(pl.col('datetime')).sort('datetime').head(1).item()
                last_time = daily_raw.select(pl.col('datetime')).sort('datetime').tail(1).item()
                print(f"\n⏰ Raw time range: {first_time} to {last_time}")
                
                # Check if last_time is before 14:00 (this would indicate the bug is in raw data)
                if datetime_dtype == pl.Datetime:
                    last_hour = last_time.hour if hasattr(last_time, 'hour') else None
                else:
                    last_hour = int(str(last_time)[11:13]) if len(str(last_time)) > 13 else None
                
                if last_hour and last_hour < 14:
                    print(f"🚨 BUG DETECTED: Raw data stops at hour {last_hour}, should continue beyond 14:00!")
                elif last_hour and last_hour >= 14:
                    print(f"✅ Raw data extends to hour {last_hour}, bug must be in processing pipeline")
                
                # Show last few records
                print("\n📋 Last 5 raw records:")
                last_raw = daily_raw.sort('datetime').tail(5)
                for row in last_raw.iter_rows(named=True):
                    dt_str = str(row['datetime'])
                    high = row['high']
                    print(f"  {dt_str}: high={high:.6f}")
        else:
            print("❌ No NOVO data found in raw file")
            available_tickers = raw_df.select('ticker').unique().head(10).to_series().to_list()
            print(f"Available tickers: {available_tickers}")
            return
            
    except Exception as e:
        print(f"❌ Error loading raw data: {e}")
        return

    # === STEP 2: Check after datetime standardization ===
    print("\n=== STEP 2: After datetime standardization ===")
    try:
        standardized_df = standardizer.standardize_to_gmt(raw_df)
        print(f"✓ Data standardized to GMT: {len(standardized_df)} rows")
        
        novo_std = standardized_df.filter(pl.col('ticker') == 'NOVOb:xcse')
        if len(novo_std) > 0:
            print(f"✓ NOVO data after standardization: {len(novo_std)} records")
            
            # Use most recent date
            max_date = novo_std.select(pl.col('datetime').dt.date().max()).item()
            print(f"\n🔍 Checking standardized data on {max_date}:")
            
            daily_std = novo_std.filter(pl.col('datetime').dt.date() == max_date)
            print(f"Standardized records on {max_date}: {len(daily_std)}")
            
            if len(daily_std) > 0:
                # Show hourly distribution after standardization
                hourly_std = daily_std.with_columns(
                    pl.col('datetime').dt.hour().alias('hour')
                ).group_by('hour').agg(pl.len().alias('count')).sort('hour')
                
                print("\n📊 After GMT conversion hourly distribution:")
                for row in hourly_std.iter_rows(named=True):
                    print(f"  Hour {row['hour']:02d}: {row['count']} records")
                
                # Show time range
                first_dt = daily_std.select(pl.col('datetime').min()).item()
                last_dt = daily_std.select(pl.col('datetime').max()).item()
                print(f"\n⏰ Standardized time range: {first_dt} to {last_dt}")
                
                # Show last few records
                print("\n📋 Last 5 standardized records:")
                last_std = daily_std.sort('datetime').tail(5)
                for row in last_std.iter_rows(named=True):
                    print(f"  {row['datetime']}: high={row['high']:.6f}")
        else:
            print("❌ No NOVO data after standardization")
            return
            
    except Exception as e:
        print(f"❌ Error in standardization: {e}")
        return

    # === STEP 3: Check after complete timeline creation ===
    print("\n=== STEP 3: After complete timeline creation ===")
    try:
        print("🔧 Creating complete timeline for NOVO...")
        novo_timeline = standardizer.create_complete_timeline(standardized_df, 'NOVOb:xcse')
        print(f"✓ Timeline created: {len(novo_timeline)} total slots")
        
        # Check same date
        daily_timeline = novo_timeline.filter(pl.col('datetime').dt.date() == max_date)
        print(f"\nTimeline slots on {max_date}: {len(daily_timeline)}")
        
        if len(daily_timeline) > 0:
            # Show hourly distribution after timeline creation
            hourly_timeline = daily_timeline.with_columns(
                pl.col('datetime').dt.hour().alias('hour')
            ).group_by('hour').agg([
                pl.len().alias('total_slots'),
                pl.col('high').is_not_null().sum().alias('data_points')
            ]).sort('hour')
            
            print("\n📊 After timeline creation hourly distribution:")
            for row in hourly_timeline.iter_rows(named=True):
                total = row['total_slots']
                data = row['data_points']
                print(f"  Hour {row['hour']:02d}: {data}/{total} data points ({data/total*100:.1f}%)")
            
            # Find where data actually stops
            last_data_record = daily_timeline.filter(pl.col('high').is_not_null()).sort('datetime').tail(1)
            if len(last_data_record) > 0:
                last_data_time = last_data_record.select('datetime').item()
                print(f"\n🛑 Data actually stops at: {last_data_time}")
            else:
                print("\n❌ No actual data found in timeline!")
            
            # Timeline range
            timeline_start = daily_timeline.select(pl.col('datetime').min()).item()
            timeline_end = daily_timeline.select(pl.col('datetime').max()).item()
            print(f"📅 Timeline range: {timeline_start} to {timeline_end}")
            
            # Show last few records with actual data
            last_data = daily_timeline.filter(pl.col('high').is_not_null()).sort('datetime').tail(5)
            if len(last_data) > 0:
                print("\n📋 Last 5 records with actual data:")
                for row in last_data.iter_rows(named=True):
                    print(f"  {row['datetime']}: high={row['high']:.6f}")
            
            # Show what happens around 13:00 specifically
            print("\n🔍 Examining 12:30-13:30 period:")
            midday_data = daily_timeline.filter(
                (pl.col('datetime').dt.hour() >= 12) & 
                (pl.col('datetime').dt.hour() <= 13)
            ).sort('datetime')
            
            for row in midday_data.iter_rows(named=True):
                dt = row['datetime']
                high = row['high']
                high_str = f"{high:.6f}" if high is not None else "NULL"
                print(f"  {dt.strftime('%H:%M')}: {high_str}")
        else:
            print("❌ No timeline data found")
            
    except Exception as e:
        print(f"❌ Error in timeline creation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("🔍 Debug analysis complete!")
    print("\nLook for patterns in the output above:")
    print("1. Does raw data extend beyond 13:00?")
    print("2. Does GMT conversion change the data range?") 
    print("3. Does timeline creation cut off data?")
    print("4. Are there gaps in the 12:30-13:30 period?")

if __name__ == "__main__":
    main()