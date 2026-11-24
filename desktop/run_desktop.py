"""
Boali Hospital Specialized Radiology Suite - Desktop Entry Point
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (نگارش v3.0.2)
"""

import os
import sys

# Ensure project root in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

offline_dir = os.path.join(root_dir, "offline_package")
if offline_dir not in sys.path:
    sys.path.insert(0, offline_dir)

if __name__ == "__main__":
    try:
        import BoneSuppression_Desktop
        BoneSuppression_Desktop.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
