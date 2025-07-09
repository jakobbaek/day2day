#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

try:
    print("Testing basic imports...")
    
    # Test basic package import
    import day2day
    print("✓ day2day package imported successfully")
    
    # Test config import
    from day2day.config.settings import settings
    print("✓ settings imported successfully")
    
    # Test data imports
    from day2day.data.auth import SaxoAuthenticator
    print("✓ SaxoAuthenticator imported successfully")
    
    from day2day.data.market_data import MarketDataCollector
    print("✓ MarketDataCollector imported successfully")
    
    # Test API imports
    from day2day.api.main import Day2DayAPI
    print("✓ Day2DayAPI imported successfully")
    
    from day2day.api.cli import main
    print("✓ CLI main imported successfully")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()