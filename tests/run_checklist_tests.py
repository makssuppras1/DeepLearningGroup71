"""
Test runner for the testing checklist.

This script runs all tests related to the testing checklist:
- All activation functions return correct shapes
- Derivatives are numerically correct
- Loss decreases with correct predictions
- Initializations have correct variance
"""
import sys
import os
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_checklist_tests():
    """Run all checklist tests and report results."""
    print("=" * 70)
    print("Running Testing Checklist Tests")
    print("=" * 70)
    print()
    
    test_files = [
        'tests/test_activations.py::TestActivationShapes',
        'tests/test_derivatives_numerical.py',
        'tests/test_loss_behavior.py',
        'tests/test_initializers.py'
    ]
    
    results = {}
    
    for test_file in test_files:
        print(f"\n{'=' * 70}")
        print(f"Running: {test_file}")
        print('=' * 70)
        
        try:
            exit_code = pytest.main([
                test_file,
                '-v',
                '--tb=short'
            ])
            results[test_file] = exit_code == 0
        except Exception as e:
            print(f"Error running {test_file}: {e}")
            results[test_file] = False
    
    print("\n" + "=" * 70)
    print("Testing Checklist Summary")
    print("=" * 70)
    print()
    
    checklist_items = [
        ("All activation functions return correct shapes", 
         'tests/test_activations.py::TestActivationShapes'),
        ("Derivatives are numerically correct", 
         'tests/test_derivatives_numerical.py'),
        ("Loss decreases with correct predictions", 
         'tests/test_loss_behavior.py'),
        ("Initializations have correct variance", 
         'tests/test_initializers.py')
    ]
    
    all_passed = True
    for item_name, test_file in checklist_items:
        status = "✓ PASS" if results.get(test_file, False) else "✗ FAIL"
        print(f"[{status}] {item_name}")
        if not results.get(test_file, False):
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checklist items passed!")
    else:
        print("⚠️  Some checklist items failed. Please review the test output above.")
    
    return all_passed


if __name__ == '__main__':
    success = run_checklist_tests()
    sys.exit(0 if success else 1)

