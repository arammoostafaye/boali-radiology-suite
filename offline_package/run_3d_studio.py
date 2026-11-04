"""
Boali Hospital 3D CT Studio - Standalone Launcher
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
نسخه v3.0.0
"""

import os
import sys
import tkinter as tk

# Ensure current dir in path
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from boali_ct_3d_studio_dialog import Boali3DCTStudioDialog


def main():
    root = tk.Tk()
    root.withdraw() # Hide root window, show dedicated 3D Studio
    dialog = Boali3DCTStudioDialog(root)
    dialog.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
