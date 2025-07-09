# Target Prediction Update: HIGH Price Only

## Summary

As per updated specifications, the day2day application now **always predicts the HIGH price** of the target instrument, regardless of any `target_price_type` parameter provided.

## Changes Made

### 1. Data Preparation (`day2day/data/preparation.py`)
- **Default parameter**: `target_price_type` default changed from `"close"` to `"high"`
- **Forced target**: The method now always forces `target_price_type = "high"` internally
- **Documentation**: Added clear notes about the requirement
- **Logging**: Added informative log message about using HIGH price

```python
# Create target variable - always use high price as per specifications
target_price_type = "high"  # Force high price as target
target_column = f"{target_price_type}_{target_instrument}"
logger.info(f"Creating target variable using HIGH price: {target_column}")
```

### 2. API Interface (`day2day/api/main.py`)
- **Parameter default**: Changed `target_price_type` default to `"high"`
- **Documentation**: Added note about HIGH price requirement
- **Logging**: Added informative log message during model training

### 3. CLI Interface (`day2day/api/cli.py`)
- **Parameter default**: Changed `target_price_type` default to `"high"`
- **Help text**: Updated to indicate HIGH price is always used
- **Output messages**: Added confirmation that HIGH price is being used

### 4. Model Training (`day2day/models/training.py`)
- **Documentation**: Updated docstrings to clarify HIGH price prediction
- **Logging**: Added log message confirming HIGH price target

### 5. Documentation (`CLAUDE.md`)
- **Target Instrument section**: Added important note about HIGH price requirement
- **Usage examples**: Updated to reflect HIGH price prediction

## Usage Examples

### CLI Usage
```bash
# Prepare training data (always predicts HIGH price)
day2day prepare --raw-data-file danish_stocks_1m.csv --output-title my_data --target-instrument NOVO-B.CO

# Output will show:
# ✓ Target prediction: HIGH price of target instrument (as per specifications)
```

### API Usage
```python
# Prepare training data
api.prepare_training_data(
    raw_data_file="danish_stocks_1m.csv",
    output_title="my_data",
    target_instrument="NOVO-B.CO"
    # target_price_type is ignored - always uses "high"
)

# Train models
api.train_models(
    training_data_title="my_data",
    target_instrument="NOVO-B.CO"
)
```

## Key Points

### 1. **Always HIGH Price**
- Regardless of the `target_price_type` parameter provided
- System internally forces the target to be the HIGH price
- Clear logging and documentation about this behavior

### 2. **Backward Compatibility**
- The `target_price_type` parameter still exists to maintain API compatibility
- However, it's now effectively ignored
- No breaking changes to existing code

### 3. **Clear Communication**
- CLI shows confirmation message about HIGH price usage
- API logs indicate HIGH price prediction
- Documentation clearly states the requirement

### 4. **Prediction Horizon**
- Still predicts HIGH price 2 hours into the future
- Uses the same 24 5-minute interval horizon
- All existing model training and evaluation logic remains intact

## Technical Details

### Target Variable Creation
The system now creates the target variable as:
```python
target_column = f"high_{target_instrument}"
```

For example, if `target_instrument="NOVO-B.CO"`, the target column will be `"high_NOVO-B.CO"`.

### Model Training
All models in the suite will predict the HIGH price of the target instrument at the 2-hour horizon.

### Feature Engineering
All other features (low, open, close prices, percentage changes, historical means, etc.) remain available as inputs to the models.

### Evaluation and Backtesting
- Model evaluation compares predicted HIGH prices to actual HIGH prices
- Backtesting strategies use HIGH price predictions for decision making
- All existing evaluation metrics remain valid

## Migration Guide

### For Existing Users
1. **No code changes required**: Existing code will continue to work
2. **Behavior change**: Models will now predict HIGH price instead of close price
3. **Verification**: Check logs to confirm HIGH price usage

### For New Users
1. **Default behavior**: HIGH price prediction is automatic
2. **No configuration needed**: System handles this requirement internally
3. **Focus on instrument selection**: Choose appropriate target instrument

## Benefits

### 1. **Consistency**
- All models predict the same target type (HIGH price)
- Eliminates confusion about prediction target
- Standardizes model comparison

### 2. **Specification Compliance**
- Meets the exact requirement from updated specifications
- Ensures correct model behavior
- Maintains system integrity

### 3. **Trading Relevance**
- HIGH price is relevant for trading decisions
- Useful for take-profit and stop-loss calculations
- Aligns with trading strategy requirements

## Verification

To verify the system is using HIGH price:

1. **Check CLI output**: Look for "✓ Target prediction: HIGH price of target instrument"
2. **Check logs**: Look for "Creating target variable using HIGH price"
3. **Inspect data**: Target column will be named `high_{instrument}`
4. **Model evaluation**: Predictions will be compared against HIGH prices

## Future Considerations

- The requirement is now hardcoded into the system
- Any future changes to target type would require code modifications
- Consider making this configurable if requirements change
- All downstream systems (evaluation, backtesting) are compatible with HIGH price prediction