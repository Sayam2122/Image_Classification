"""
Auto-classify script - automatically loads model and classifies default image
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Backup stdin and create fake input
import sys
from io import StringIO

# Simulate user inputs
fake_input = StringIO("y\n\n")
original_stdin = sys.stdin
sys.stdin = fake_input

try:
    # Import and run main
    import satellite_classifier_hierarchical
    satellite_classifier_hierarchical.main()
finally:
    # Restore stdin
    sys.stdin = original_stdin
