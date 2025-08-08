"""
Main entry point for PhantomTouch.

This script delegates to the modular application in `phantom_touch.app`.
Run `python main.py` or `python -m phantom_touch.app`.
"""

from phantom_touch.app import main

if __name__ == "__main__":
    main()


