# Setup Guide for day2day

## Resolving Dependency Conflicts

The conflicts you're experiencing are due to version incompatibilities between newer numpy/scipy versions and older packages like `numba` and `pymc3`.

## Solution Options

### Option 1: Clean Installation (Recommended)

1. **Create a new virtual environment:**
   ```bash
   python -m venv day2day_env
   source day2day_env/bin/activate  # On Windows: day2day_env\Scripts\activate
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   day2day status
   ```

### Option 2: Fix Existing Environment

1. **Uninstall conflicting packages:**
   ```bash
   pip uninstall numba pymc3 numpy scipy -y
   ```

2. **Install compatible versions:**
   ```bash
   pip install numpy==1.24.4 scipy==1.7.3
   ```

3. **Reinstall the package:**
   ```bash
   pip install -e .
   ```

### Option 3: Use conda (Alternative)

If you prefer conda, create an environment with compatible versions:

```bash
conda create -n day2day python=3.9
conda activate day2day
conda install numpy=1.24 scipy=1.7
pip install -e .
```

## Package Version Constraints

The updated `requirements.txt` now includes:
- `numpy>=1.21.0,<1.25.0` (compatible with numba 0.57.1)
- `scipy>=1.7.3,<1.8.0` (compatible with pymc3 3.11.5)
- `pandas>=1.5.0,<2.0.0` (maintains stability)

## Optional Dependencies

Some packages (`numba`, `pymc3`) have been commented out because they're not core requirements for the day2day application. If you need them for specific features:

1. **Install them separately:**
   ```bash
   pip install numba==0.57.1 pymc3==3.11.5
   ```

2. **Or create a separate environment for advanced analytics:**
   ```bash
   python -m venv advanced_env
   source advanced_env/bin/activate
   pip install numba pymc3 pystan theano
   ```

## Core day2day Features

The application works perfectly with the core dependencies:
- Market data collection via Saxo Bank API
- Data preparation and feature engineering
- Model training (XGBoost, Random Forest, Linear models)
- Bootstrapping and uncertainty estimation
- Model evaluation and visualization
- Backtesting with trading strategies

## Troubleshooting

If you still encounter issues:

1. **Check your Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

3. **Use pip-tools for dependency resolution:**
   ```bash
   pip install pip-tools
   pip-compile requirements.txt
   pip-sync requirements.txt
   ```

4. **Check for conflicting packages:**
   ```bash
   pip check
   ```

## Environment File Setup

Don't forget to set up your environment variables:

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API credentials:**
   ```bash
   SAXO_CLIENT_ID=your_client_id
   SAXO_CLIENT_SECRET=your_client_secret
   SAXO_ACCESS_TOKEN=your_access_token
   ```

## Next Steps

After resolving the dependencies:

1. **Test the installation:**
   ```bash
   python main.py
   ```

2. **Run the CLI:**
   ```bash
   day2day --help
   ```

3. **Start with data collection:**
   ```bash
   day2day collect --start-date 2023-01-01 --end-date 2023-12-31
   ```