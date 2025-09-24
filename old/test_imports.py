# Simple test script for the modular system
import sys
import os

def test_imports():
    """Test if all modules can be imported"""
    try:
        print("Testing imports...")
        
        import config
        print("✅ config imported")
        
        import data_prep
        print("✅ data_prep imported")
        
        import technical_indicators
        print("✅ technical_indicators imported")
        
        import time_series_models
        print("✅ time_series_models imported")
        
        import volatility_models
        print("✅ volatility_models imported")
        
        import forecasting
        print("✅ forecasting imported")
        
        import plotting
        print("✅ plotting imported")
        
        import pdf_generation
        print("✅ pdf_generation imported")
        
        print("\n🎉 All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ All modules are ready!")
    else:
        print("\n❌ Some modules have issues!")
        sys.exit(1)
