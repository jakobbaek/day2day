# Datetime Standardization in day2day

## Overview

The day2day application now includes comprehensive datetime standardization to ensure consistent 1-minute granularity data across all instruments during market hours. This addresses the requirement that "not all instruments will have price data for each time step of the 1-minute granularity during market opening hours."

## Key Features

### 1. GMT Timezone Conversion
- All timestamps are converted to Greenwich Mean Time (GMT/UTC)
- Handles Danish timezone conversion (CET/CEST) automatically
- Accounts for daylight saving time transitions

### 2. Market Hours Inference
- Automatically detects market opening hours from the data itself
- Infers hours per instrument based on actual trading activity
- Caches market hours for performance
- Falls back to Copenhagen Stock Exchange defaults (8:00-16:00 GMT)

### 3. Complete Timeline Creation
- Generates complete 1-minute intervals for all market hours
- Fills missing data points with forward-filled prices
- Ensures every instrument has data for every minute during market hours
- Skips weekends automatically

### 4. Forward Fill Strategy
- When an instrument lacks price data for a specific minute, the last available price is used
- Maintains price continuity during market hours
- Handles gaps in data seamlessly

## Implementation Details

### DateTimeStandardizer Class

The `DateTimeStandardizer` class in `day2day/data/datetime_utils.py` provides:

```python
class DateTimeStandardizer:
    def standardize_to_gmt(self, df: pl.DataFrame) -> pl.DataFrame
    def infer_market_hours(self, df: pl.DataFrame, ticker: str) -> Tuple[pd.Time, pd.Time]
    def create_complete_timeline(self, df: pl.DataFrame, ticker: str) -> pl.DataFrame
    def standardize_all_instruments(self, df: pl.DataFrame) -> pl.DataFrame
```

### Usage in Data Preparation

The standardization is integrated into the data preparation pipeline:

```python
# Automatic standardization (default)
df = data_preparator.load_raw_data("danish_stocks_1m.csv", standardize_datetime=True)

# Disable standardization if needed
df = data_preparator.load_raw_data("danish_stocks_1m.csv", standardize_datetime=False)
```

## CLI Usage

### Enable Standardization (Default)
```bash
day2day prepare --raw-data-file danish_stocks_1m.csv --output-title my_data --target-instrument NOVO-B.CO
```

### Disable Standardization
```bash
day2day prepare --raw-data-file danish_stocks_1m.csv --output-title my_data --target-instrument NOVO-B.CO --no-standardize-datetime
```

## Process Flow

1. **Load Raw Data**: Read CSV file with original timestamps
2. **GMT Conversion**: Convert Danish timezone (CET/CEST) to GMT
3. **Market Hours Inference**: Analyze data to determine typical trading hours per instrument
4. **Timeline Generation**: Create complete 1-minute timeline for market hours
5. **Forward Fill**: Fill missing price data using last available values
6. **Quality Filtering**: Remove instruments with insufficient data quality

## Benefits

### Data Quality
- Ensures consistent temporal resolution across all instruments
- Eliminates gaps in trading data
- Standardizes timezone handling

### Model Training
- Provides clean, regular time series data
- Reduces noise from irregular sampling
- Enables better feature engineering

### Backtesting
- Accurate simulation of real-time trading conditions
- Proper handling of market hours and data availability
- Realistic price progression during gaps

## Configuration Options

### Environment Variables
```bash
# Default timezone handling
DEFAULT_EXCHANGE=CSE
PREDICTION_HORIZON_HOURS=2

# Data quality thresholds
MIN_DAILY_OBSERVATIONS=25
```

### API Parameters
```python
# Full control over standardization
api.prepare_training_data(
    raw_data_file="danish_stocks_1m.csv",
    output_title="my_data",
    target_instrument="NOVO-B.CO",
    standardize_datetime=True,  # Enable/disable standardization
    min_daily_observations=25   # Quality threshold
)
```

## Market Hours Detection

The system automatically detects market hours by:

1. **Analyzing Trading Activity**: Examining actual data points per instrument
2. **Statistical Analysis**: Using median opening/closing times
3. **Quality Filtering**: Excluding days with insufficient observations
4. **Caching**: Storing detected hours for performance

### Example Market Hours Detection
```
INFO: Inferring market hours for NOVO-B.CO
INFO: Inferred market hours for NOVO-B.CO: 08:00 - 16:00 GMT
INFO: Inferring market hours for CARL-B.CO
INFO: Inferred market hours for CARL-B.CO: 08:00 - 16:00 GMT
```

## Timezone Handling

### Danish Market (CET/CEST)
- **Winter (CET)**: UTC+1 → GMT = Local - 1 hour
- **Summer (CEST)**: UTC+2 → GMT = Local - 2 hours
- **DST Transitions**: Last Sunday in March/October

### Example Conversion
```
Local Time (CET):  09:00  → GMT: 08:00
Local Time (CEST): 09:00  → GMT: 07:00
```

## Performance Considerations

### Caching
- Market hours are cached per instrument
- Reduces repeated calculations
- Improves processing speed for large datasets

### Memory Usage
- Complete timelines increase data size
- Trade-off between completeness and memory
- Configurable quality thresholds

## Troubleshooting

### Common Issues

1. **Insufficient Data Quality**
   - Solution: Adjust `min_daily_observations` parameter
   - Lower threshold for instruments with sparse data

2. **Incorrect Market Hours**
   - Solution: Manual override in configuration
   - Check data quality for specific instruments

3. **Memory Issues with Large Datasets**
   - Solution: Process instruments in batches
   - Disable standardization for very large datasets

### Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Check market hours detection
standardizer = DateTimeStandardizer()
market_open, market_close = standardizer.infer_market_hours(df, "NOVO-B.CO")
print(f"Market hours: {market_open} - {market_close}")
```

## Migration Guide

### From Legacy Code
The old `fill_missing_timestamps()` method is deprecated. Use the new standardization:

```python
# Old approach (deprecated)
df = data_preparator.fill_missing_timestamps(df, reference_ticker)

# New approach (recommended)
df = data_preparator.load_raw_data("data.csv", standardize_datetime=True)
```

### Backward Compatibility
- Legacy methods still work with deprecation warnings
- Gradual migration recommended
- New projects should use standardization by default

## Future Enhancements

- Support for multiple exchanges and timezones
- Holiday calendar integration
- Advanced gap-filling strategies
- Real-time data standardization
- Performance optimizations for very large datasets