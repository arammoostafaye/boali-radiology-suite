"""
Boali Hospital Radiology AI Platform - Auto Update Module
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
سیستم بررسی و به‌روزرسانی هوشمند خودکار از مخزن رسمی گیت‌هاب
"""

import os
import re
import sys
import json
import time
import zipfile
import threading
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox

GITHUB_REPO = "arammoostafaye/boali-radiology-suite"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

def parse_version_tuple(v_str):
    """Convert 'v3.0.3', '3.0.3-beta', or '3.0.2.1' into tuple (3, 0, 2)."""
    clean = str(v_str).strip().lower().lstrip("v")
    # extract numeric digit groups
    nums = re.findall(r'\d+', clean)
    parts = [int(n) for n in nums]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

class BoaliAutoUpdater:
    def __init__(self, current_version="3.0.3", root_app=None):
        self.current_version = current_version
        self.root_app = root_app
        self.is_checking = False
        self.download_in_progress = False

    def check_for_updates_async(self, silent=True, on_complete=None):
        """Checks for new release in background thread."""
        if self.is_checking:
            return
        self.is_checking = True
        threading.Thread(
            target=self._check_worker,
            args=(silent, on_complete),
            daemon=True
        ).start()

    def _check_worker(self, silent, on_complete):
        try:
            req = urllib.request.Request(
                GITHUB_API_URL,
                headers={"User-Agent": "Boali-Radiology-Suite-Desktop"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tag_name = data.get("tag_name", "")
            remote_ver = parse_version_tuple(tag_name)
            local_ver = parse_version_tuple(self.current_version)

            has_update = remote_ver > local_ver
            release_notes = data.get("body", "به‌روزرسانی جامع پلتفرم هوش مصنوعی رادیولوژی و استودیوی سه‌بعدی.")
            
            # Find the update patch zip asset
            update_asset_url = None
            assets = data.get("assets", [])
            # Priority 1: Patch or Update zip
            for asset in assets:
                name = asset.get("name", "")
                if "Patch.zip" in name or "Update.zip" in name or "Update-Only.zip" in name:
                    update_asset_url = asset.get("browser_download_url")
                    break
            # Priority 2: Any zip if not found
            if not update_asset_url:
                for asset in assets:
                    if asset.get("name", "").endswith(".zip"):
                        update_asset_url = asset.get("browser_download_url")
                        break

            if on_complete:
                if self.root_app:
                    self.root_app.after(0, lambda: on_complete(has_update, tag_name, release_notes, update_asset_url))
                else:
                    on_complete(has_update, tag_name, release_notes, update_asset_url)
            elif has_update:
                if self.root_app:
                    self.root_app.after(0, lambda: self.show_update_dialog(tag_name, release_notes, update_asset_url))
            elif not silent:
                if self.root_app:
                    self.root_app.after(0, lambda: messagebox.showinfo(
                        "بررسی به‌روزرسانی",
                        f"شما در حال استفاده از آخرین نگارش نرم‌افزار بوعلی (v{self.current_version}) هستید.\nهیچ آپدیت جدیدی در مخزن وجود ندارد."
                    ))

        except Exception as e:
            if not silent:
                if self.root_app:
                    self.root_app.after(0, lambda: messagebox.showwarning(
                        "بررسی به‌روزرسانی",
                        f"امکان برقراری ارتباط با مخزن گیت‌هاب فراهم نشد:\n{e}"
                    ))
        finally:
            self.is_checking = False

    def show_update_dialog(self, new_version, notes, download_url):
        """Displays a dedicated dialog asking user to apply update."""
        parent = self.root_app or tk._default_root
        dlg = tk.Toplevel(parent)
        dlg.title("به‌روزرسانی جدید موجود است | نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی")
        dlg.geometry("560x420")
        dlg.resizable(False, False)
        dlg.configure(bg="#0B0F19")
        dlg.grab_set()

        # Header
        hdr = tk.Frame(dlg, bg="#111827", height=65)
        hdr.pack(fill=tk.X)

        tk.Label(
            hdr,
            text=f"🚀 نگارش جدید {new_version} آماده دریافت است!",
            font=("Vazirmatn", 11, "bold"),
            fg="#38BDF8",
            bg="#111827"
        ).pack(anchor=tk.E, padx=16, pady=(12, 2))

        tk.Label(
            hdr,
            text=f"نسخه فعلی شما: v{self.current_version} | نسخه منتشرشده در مخزن: {new_version}",
            font=("Vazirmatn", 8),
            fg="#94A3B8",
            bg="#111827"
        ).pack(anchor=tk.E, padx=16, pady=(0, 10))

        # Body - Release notes
        body = tk.Frame(dlg, bg="#0B0F19", padx=16, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            body,
            text="تغییرات و قابلیت‌های اضافه شده در این نگارش:",
            font=("Vazirmatn", 9, "bold"),
            fg="#E2E8F0",
            bg="#0B0F19"
        ).pack(anchor=tk.E, pady=(0, 6))

        txt_box = tk.Text(body, bg="#1E293B", fg="#F1F5F9", font=("Vazirmatn", 8),
                          wrap=tk.WORD, height=9, bd=0, padx=8, pady=8)
        txt_box.pack(fill=tk.BOTH, expand=True)
        txt_box.insert(tk.END, notes)
        txt_box.config(state=tk.DISABLED)

        # Progress bar
        prog_bar = ttk.Progressbar(body, orient=tk.HORIZONTAL, mode='determinate')
        lbl_status = tk.Label(body, text="", font=("Vazirmatn", 8), fg="#34D399", bg="#0B0F19")

        # Action Buttons
        btn_box = tk.Frame(dlg, bg="#111827", height=50)
        btn_box.pack(fill=tk.X, side=tk.BOTTOM)

        def do_update():
            if not download_url:
                messagebox.showerror("خطا", "آدرس دانلود مستقیم پکیج آپدیت یافت نشد.")
                return
            btn_update.config(state=tk.DISABLED, bg="#334155")
            btn_cancel.config(state=tk.DISABLED)
            prog_bar.pack(fill=tk.X, pady=(10, 4))
            lbl_status.pack(anchor=tk.E)
            lbl_status.config(text="در حال دریافت پکیج سبک به‌روزرسانی از گیت‌هاب...")

            threading.Thread(
                target=self._download_and_extract_patch,
                args=(download_url, dlg, prog_bar, lbl_status),
                daemon=True
            ).start()

        btn_cancel = tk.Button(
            btn_box,
            text="بعداً یادآوری کن",
            font=("Vazirmatn", 9),
            bg="#334155",
            fg="#E2E8F0",
            relief=tk.FLAT,
            padx=14,
            pady=5,
            cursor="hand2",
            command=dlg.destroy
        )
        btn_cancel.pack(side=tk.LEFT, padx=16, pady=10)

        btn_update = tk.Button(
            btn_box,
            text="✓ دریافت و به‌روزرسانی خودکار",
            font=("Vazirmatn", 9, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=16,
            pady=5,
            cursor="hand2",
            command=do_update
        )
        btn_update.pack(side=tk.RIGHT, padx=16, pady=10)

    def _download_and_extract_patch(self, url, dlg, prog_bar, lbl_status):
        try:
            exe_dir = os.path.dirname(os.path.abspath(__file__))
            temp_zip = os.path.join(exe_dir, "temp_update_patch.zip")

            req = urllib.request.Request(url, headers={"User-Agent": "Boali-Radiology-Suite-Desktop"})
            with urllib.request.urlopen(req) as resp, open(temp_zip, "wb") as out_f:
                total_len = resp.getheader('content-length')
                total_bytes = int(total_len) if total_len else 1000000
                dl = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    dl += len(chunk)
                    out_f.write(chunk)
                    pct = int(dl * 100 / total_bytes)
                    prog_bar["value"] = pct
                    lbl_status.config(text=f"در حال دانلود: {dl // 1024} KB / {total_bytes // 1024} KB ({pct}%)")

            lbl_status.config(text="در حال استخراج و جایگزینی فایل‌های به‌روزرسانی...")
            time.sleep(0.5)

            with zipfile.ZipFile(temp_zip, "r") as z:
                z.extractall(exe_dir)

            if os.path.exists(temp_zip):
                os.remove(temp_zip)

            lbl_status.config(text="✓ به‌روزرسانی با موفقیت انجام شد! در حال راه‌اندازی مجدد...")
            time.sleep(1.0)

            # Restart app
            messagebox.showinfo(
                "به‌روزرسانی موفق",
                "نرم‌افزار با موفقیت به‌روزرسانی شد.\nبرنامه جهت بارگذاری تغییرات مجدداً راه‌اندازی می‌شود."
            )
            dlg.destroy()
            if self.root_app:
                self.root_app.destroy()
            python_exe = sys.executable
            main_script = os.path.join(exe_dir, "BoneSuppression_Desktop.py")
            os.execl(python_exe, python_exe, main_script)

        except Exception as e:
            lbl_status.config(text=f"خطا در به‌روزرسانی: {e}", fg="#F87171")
            messagebox.showerror("خطای به‌روزرسانی", f"فرآیند دانلود یا نصب خودکار با خطا مواجه شد:\n{e}")


# ---------------------------------------------------------------------------
# Convenient Module-Level Functions
# ---------------------------------------------------------------------------
_global_updater = None

def get_updater(root_app=None, current_version="3.0.3"):
    global _global_updater
    if _global_updater is None:
        _global_updater = BoaliAutoUpdater(current_version=current_version, root_app=root_app)
    else:
        if root_app:
            _global_updater.root_app = root_app
        if current_version:
            _global_updater.current_version = current_version
    return _global_updater

def check_updates_manual(root_app, current_version="3.0.3"):
    """بررسی دستی به‌روزرسانی توسط کاربر با نمایش پیام در صورت نبود آپدیت"""
    updater = get_updater(root_app, current_version)
    updater.check_for_updates_async(silent=False)

def start_background_update_check(root_app, current_version="3.0.3", delay_sec=4):
    """بررسی بی‌صدا و نامحسوس آپدیت در پس‌زمینه بعد از اجرای برنامه"""
    def _delayed():
        updater = get_updater(root_app, current_version)
        updater.check_for_updates_async(silent=True)
    if hasattr(root_app, "after"):
        root_app.after(int(delay_sec * 1000), _delayed)
    else:
        import threading
        threading.Timer(delay_sec, _delayed).start()
