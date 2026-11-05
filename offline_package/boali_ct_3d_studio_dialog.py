"""
Boali Hospital Radiology AI Platform - 3D CT Reconstruction Studio Dialog
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
نسخه v3.0.0

Interactive Desktop Studio for Clinical 3D Volume Rendering, 360-degree Cine, and Multi-Organ CT Presets.
Developer: آرام مصطفائی (Aram Mostafaei)
Network: شبکه بهداشت و درمان مریوان - بیمارستان بوعلی
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Ensure current directory is in sys.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

try:
    from boali_ct_3d_engine import Boali3DReconstructor, PRESETS
except Exception as err:
    try:
        import importlib.util
        engine_path = os.path.join(cur_dir, "boali_ct_3d_engine.py")
        if os.path.exists(engine_path):
            spec = importlib.util.spec_from_file_location("boali_ct_3d_engine", engine_path)
            engine_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine_mod)
            Boali3DReconstructor = engine_mod.Boali3DReconstructor
            PRESETS = engine_mod.PRESETS
        else:
            raise err
    except Exception:
        raise err

# Theme tokens
C_CANVAS = "#090d16"
C_CARD = "#111827"
C_HEADER = "#0f172a"
C_BORDER = "#27354a"
C_CYAN = "#38bdf8"
C_BLUE = "#2563eb"
C_GREEN = "#059669"
C_TEXT_LIGHT = "#f8fafc"
C_TEXT_MUTED = "#94a3b8"


class Boali3DCTStudioDialog(tk.Toplevel):
    """Clinical 3D CT Reconstruction Studio Window."""

    def __init__(self, parent, default_folder=None, report_callback=None):
        super().__init__(parent)
        self.title("استودیوی بازسازی سه‌بعدی سی‌تی اسکن | نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (v3.0.0)")
        self.geometry("1180x780")
        self.minsize(1000, 680)
        self.configure(bg=C_CANVAS)

        self.engine = Boali3DReconstructor(target_resolution=(256, 256))
        self.default_folder = default_folder
        self.report_callback = report_callback

        self.loaded_slices_count = 0
        self.cine_frames = []
        self.current_frame_idx = 0
        self.is_processing = False

        self._build_ui()

        if self.default_folder and os.path.exists(self.default_folder):
            self.folder_var.set(self.default_folder)
            self._start_reconstruction_thread()

    def _build_ui(self):
        # 1. Top Header
        top_bar = tk.Frame(self, bg=C_HEADER, height=54, highlightthickness=1, highlightbackground=C_BORDER)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        lbl_brand = tk.Label(
            top_bar,
            text="🏥 نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی | ماژول بازسازی سه‌بعدی سی‌تی اسکن (v3.0.0)",
            font=("Vazirmatn", 11, "bold"),
            fg=C_CYAN,
            bg=C_HEADER
        )
        lbl_brand.pack(side=tk.RIGHT, padx=16, pady=10)

        lbl_sub = tk.Label(
            top_bar,
            text="3D Volume Rendering & Multi-Organ Clinical CT Studio",
            font=("Arial", 9),
            fg=C_TEXT_MUTED,
            bg=C_HEADER
        )
        lbl_sub.pack(side=tk.LEFT, padx=16, pady=10)

        # 2. Main Content Splitter (Left Controls, Right 3D Viewer)
        main_frame = tk.Frame(self, bg=C_CANVAS)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # --- LEFT CONTROLS PANEL ---
        left_panel = tk.Frame(main_frame, bg=C_CARD, width=380, highlightthickness=1, highlightbackground=C_BORDER)
        left_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        left_panel.pack_propagate(False)

        # Title
        tk.Label(
            left_panel,
            text="⚙️ تنظیمات بازسازی و پریست آناتومیک",
            font=("Vazirmatn", 10, "bold"),
            fg=C_TEXT_LIGHT,
            bg=C_CARD
        ).pack(anchor=tk.E, padx=14, pady=(12, 6))

        # Folder Input
        f_box = tk.LabelFrame(left_panel, text="پوشه مقاطع سی‌تی اسکن (DICOM / Images)", font=("Vazirmatn", 8, "bold"),
                              fg=C_CYAN, bg=C_CARD, labelanchor="ne")
        f_box.pack(fill=tk.X, padx=12, pady=6)

        self.folder_var = tk.StringVar(value="")
        entry_f = tk.Entry(f_box, textvariable=self.folder_var, font=("Arial", 8), bg=C_CANVAS, fg=C_TEXT_LIGHT, insertbackground="white")
        entry_f.pack(fill=tk.X, padx=8, pady=(4, 2))

        btn_browse = tk.Button(
            f_box,
            text="📂 انتخاب پوشه اسلایس‌ها...",
            font=("Vazirmatn", 8),
            bg=C_BORDER,
            fg=C_TEXT_LIGHT,
            relief=tk.FLAT,
            command=self._browse_folder
        )
        btn_browse.pack(fill=tk.X, padx=8, pady=(2, 6))

        # Preset Selector
        p_box = tk.LabelFrame(left_panel, text="پریست تخصصی ارگان (Clinical Preset)", font=("Vazirmatn", 8, "bold"),
                              fg=C_CYAN, bg=C_CARD, labelanchor="ne")
        p_box.pack(fill=tk.X, padx=12, pady=6)

        self.preset_var = tk.StringVar(value="BONE_TRAUMA")
        self.preset_combo = ttk.Combobox(
            p_box,
            textvariable=self.preset_var,
            state="readonly",
            font=("Vazirmatn", 8)
        )
        self.preset_combo["values"] = [
            "BONE_TRAUMA - استخوان و تروما (ارتوپدی و مفاصل)",
            "CRANIOFACIAL - جمجمه، فک و صورت",
            "SPINE_VERTEBRAE - ستون فقرات و مهره‌ها",
            "PELVIS_ACETABULUM - حلقه لگن و استابولوم",
            "VASCULAR_CTA_MIP - آنژیوگرافی عروق (3D MIP)",
            "AIRWAY_BRONCHO - راه‌های هوایی و نای"
        ]
        self.preset_combo.current(0)
        self.preset_combo.pack(fill=tk.X, padx=8, pady=6)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        self.lbl_desc = tk.Label(p_box, text=PRESETS["BONE_TRAUMA"]["description"], font=("Vazirmatn", 7),
                                 fg=C_TEXT_MUTED, bg=C_CARD, wraplength=340, justify=tk.RIGHT)
        self.lbl_desc.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Density Threshold Slider
        th_box = tk.LabelFrame(left_panel, text="آستانه دانسیته هانسفیلد (HU / Threshold)", font=("Vazirmatn", 8, "bold"),
                               fg=C_CYAN, bg=C_CARD, labelanchor="ne")
        th_box.pack(fill=tk.X, padx=12, pady=6)

        self.threshold_var = tk.IntVar(value=PRESETS["BONE_TRAUMA"]["hu_threshold"])
        self.lbl_th_val = tk.Label(th_box, text=f"مقدار فعلی: {self.threshold_var.get()}", font=("Vazirmatn", 8),
                                   fg=C_TEXT_LIGHT, bg=C_CARD)
        self.lbl_th_val.pack(anchor=tk.E, padx=8, pady=(2, 0))

        scale_th = ttk.Scale(th_box, from_=40, to_=220, variable=self.threshold_var, orient=tk.HORIZONTAL,
                             command=lambda v: self.lbl_th_val.config(text=f"مقدار فعلی: {int(float(v))}"))
        scale_th.pack(fill=tk.X, padx=8, pady=4)

        # Cine Frames Count
        c_box = tk.LabelFrame(left_panel, text="تعداد فریم‌های چرخش ۳۶۰ درجه", font=("Vazirmatn", 8, "bold"),
                              fg=C_CYAN, bg=C_CARD, labelanchor="ne")
        c_box.pack(fill=tk.X, padx=12, pady=6)

        self.frames_var = tk.IntVar(value=24)
        r_frame = tk.Frame(c_box, bg=C_CARD)
        r_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Radiobutton(r_frame, text="۱۲ فریم (سریع)", variable=self.frames_var, value=12, bg=C_CARD, fg=C_TEXT_LIGHT, selectcolor=C_CANVAS, font=("Vazirmatn", 8)).pack(side=tk.RIGHT, padx=4)
        tk.Radiobutton(r_frame, text="۲۴ فریم (استاندارد)", variable=self.frames_var, value=24, bg=C_CARD, fg=C_TEXT_LIGHT, selectcolor=C_CANVAS, font=("Vazirmatn", 8)).pack(side=tk.RIGHT, padx=4)
        tk.Radiobutton(r_frame, text="۳۶ فریم (روان)", variable=self.frames_var, value=36, bg=C_CARD, fg=C_TEXT_LIGHT, selectcolor=C_CANVAS, font=("Vazirmatn", 8)).pack(side=tk.RIGHT, padx=4)

        # Action Button: Generate 3D
        self.btn_run = tk.Button(
            left_panel,
            text="🚀 شروع بازسازی ۳D و تولید سری چرخشی",
            font=("Vazirmatn", 10, "bold"),
            bg=C_BLUE,
            fg="white",
            relief=tk.FLAT,
            command=self._start_reconstruction_thread
        )
        self.btn_run.pack(fill=tk.X, padx=12, pady=(12, 6))

        # Progress
        self.progress_bar = ttk.Progressbar(left_panel, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_status = tk.Label(left_panel, text="آماده دریافت مقاطع سی‌تی اسکن...", font=("Vazirmatn", 8),
                                   fg=C_TEXT_MUTED, bg=C_CARD)
        self.lbl_status.pack(fill=tk.X, padx=12, pady=2)

        # --- RIGHT INTERACTIVE 3D VIEWER ---
        right_panel = tk.Frame(main_frame, bg=C_CARD, highlightthickness=1, highlightbackground=C_BORDER)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Viewer Header Bar
        v_head = tk.Frame(right_panel, bg=C_HEADER, height=36)
        v_head.pack(fill=tk.X, side=tk.TOP)

        self.lbl_angle = tk.Label(
            v_head,
            text="زاویه دید: ۰ درجه (نمای قدامی AP)",
            font=("Vazirmatn", 9, "bold"),
            fg=C_CYAN,
            bg=C_HEADER
        )
        self.lbl_angle.pack(side=tk.RIGHT, padx=12, pady=6)

        lbl_hint = tk.Label(
            v_head,
            text="💡 برای چرخش ۳۶۰ درجه، اسکرول موس را روی تصویر بچرخانید",
            font=("Vazirmatn", 8),
            fg=C_TEXT_MUTED,
            bg=C_HEADER
        )
        lbl_hint.pack(side=tk.LEFT, padx=12, pady=6)

        # Canvas for 3D Image Display
        self.canvas_3d = tk.Label(right_panel, bg=C_CANVAS)
        self.canvas_3d.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas_3d.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas_3d.bind("<Button-4>", lambda e: self._step_frame(-1)) # Linux scroll up
        self.canvas_3d.bind("<Button-5>", lambda e: self._step_frame(1))  # Linux scroll down

        # Bottom Slider for 360 degree rotation
        slider_frame = tk.Frame(right_panel, bg=C_HEADER, height=44)
        slider_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(slider_frame, text="چرخش ۳۶۰°:", font=("Vazirmatn", 8, "bold"), fg=C_TEXT_LIGHT, bg=C_HEADER).pack(side=tk.RIGHT, padx=(8, 4), pady=8)

        self.slider_rot = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_slider_scroll)
        self.slider_rot.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8, pady=8)

        # Bottom Export Actions
        export_bar = tk.Frame(right_panel, bg=C_CARD, height=42)
        export_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=4)

        btn_report = tk.Button(
            export_bar,
            text="📋 افزودن به گزارش بالینی بیمار",
            font=("Vazirmatn", 8, "bold"),
            bg=C_GREEN,
            fg="white",
            relief=tk.FLAT,
            command=self._export_to_report
        )
        btn_report.pack(side=tk.RIGHT, padx=6)

        btn_cine_save = tk.Button(
            export_bar,
            text="💾 ذخیره سری چرخشی در پکس / دیسک",
            font=("Vazirmatn", 8),
            bg=C_BORDER,
            fg=C_TEXT_LIGHT,
            relief=tk.FLAT,
            command=self._save_cine_series
        )
        btn_cine_save.pack(side=tk.RIGHT, padx=6)

        btn_stl = tk.Button(
            export_bar,
            text="🖨️ خروجی پرینتر سه‌بعدی ارتوپدی (.STL)",
            font=("Vazirmatn", 8),
            bg=C_BORDER,
            fg=C_TEXT_LIGHT,
            relief=tk.FLAT,
            command=self._export_stl
        )
        btn_stl.pack(side=tk.RIGHT, padx=6)

    def _browse_folder(self):
        d = filedialog.askdirectory(title="انتخاب پوشه مقاطع سی‌تی اسکن")
        if d:
            self.folder_var.set(d)

    def _on_preset_change(self, event=None):
        val = self.preset_var.get()
        p_key = val.split(" - ")[0].strip()
        if p_key in PRESETS:
            preset = PRESETS[p_key]
            self.lbl_desc.config(text=preset["description"])
            self.threshold_var.set(preset["hu_threshold"])
            self.lbl_th_val.config(text=f"مقدار فعلی: {preset['hu_threshold']}")

    def _start_reconstruction_thread(self):
        if self.is_processing:
            return
        folder = self.folder_var.get().strip()
        if not folder or not os.path.exists(folder):
            messagebox.showwarning("خطا", "لطفاً ابتدا یک پوشه معتبر حاوی مقاطع سی‌تی اسکن را انتخاب کنید.")
            return

        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, bg=C_BORDER)
        self.progress_bar["value"] = 5
        self.lbl_status.config(text="در حال بارگذاری مقاطع سی‌تی اسکن...")

        threading.Thread(target=self._run_reconstruction, daemon=True).start()

    def _run_reconstruction(self):
        try:
            folder = self.folder_var.get().strip()
            val = self.preset_var.get()
            preset_key = val.split(" - ")[0].strip()
            num_frames = self.frames_var.get()
            th = self.threshold_var.get()
            PRESETS[preset_key]["hu_threshold"] = th

            def progress_cb(pct, msg):
                self.after(0, lambda: self._update_progress(pct, msg))

            # 1. Load slices & resample
            self.engine.load_series_from_folder(folder, slice_thickness=1.25, progress_callback=progress_cb)

            # 2. Render 360 degree cine
            out_cine_dir = os.path.join(folder, "Boali_3D_Cine_Series")
            frame_paths = self.engine.generate_full_360_cine(out_cine_dir, num_frames=num_frames,
                                                            preset_key=preset_key, progress_callback=progress_cb)

            self.cine_frames = [cv2.imread(p) for p in frame_paths if os.path.exists(p)]

            self.after(0, self._on_reconstruction_done)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._on_reconstruction_error(err_msg))

    def _update_progress(self, pct, msg):
        self.progress_bar["value"] = pct
        self.lbl_status.config(text=msg)

    def _on_reconstruction_done(self):
        self.is_processing = False
        self.btn_run.config(state=tk.NORMAL, bg=C_BLUE)
        self.lbl_status.config(text="✓ بازسازی سه‌بعدی با موفقیت انجام شد.")
        if self.cine_frames:
            self.slider_rot.config(from_=0, to=len(self.cine_frames)-1)
            self.slider_rot.set(0)
            self._display_frame(0)

    def _on_reconstruction_error(self, err_msg):
        self.is_processing = False
        self.btn_run.config(state=tk.NORMAL, bg=C_BLUE)
        self.lbl_status.config(text="خطا در بازسازی ۳D")
        messagebox.showerror("خطا در بازسازی سه‌بعدی", f"عملیات با خطا مواجه شد:\n{err_msg}")

    def _display_frame(self, idx):
        if not self.cine_frames:
            return
        idx = max(0, min(idx, len(self.cine_frames) - 1))
        self.current_frame_idx = idx

        total = len(self.cine_frames)
        ang = int((idx / total) * 360)
        direction_name = "قدامی (AP)" if ang < 25 or ang > 335 else ("مایل (Oblique)" if ang < 75 or ang > 285 else ("جانبی (Lateral)" if ang < 115 or ang > 245 else "خلفی (Posterior)"))
        self.lbl_angle.config(text=f"زاویه دید: {ang} درجه | نمای {direction_name}")

        frame_bgr = self.cine_frames[idx]
        h, w = frame_bgr.shape[:2]
        # Resize to fit canvas
        canvas_h = self.canvas_3d.winfo_height() or 460
        canvas_w = self.canvas_3d.winfo_width() or 460
        scale = min(canvas_w / w, canvas_h / h, 2.0)
        nw, nh = int(w * scale), int(h * scale)
        res = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)
        self.tk_photo = ImageTk.PhotoImage(im_pil)
        self.canvas_3d.config(image=self.tk_photo)

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self._step_frame(-1)
        else:
            self._step_frame(1)

    def _step_frame(self, step):
        if not self.cine_frames:
            return
        total = len(self.cine_frames)
        new_idx = (self.current_frame_idx + step) % total
        self.slider_rot.set(new_idx)
        self._display_frame(new_idx)

    def _on_slider_scroll(self, val):
        idx = int(float(val))
        self._display_frame(idx)

    def _export_to_report(self):
        if not self.cine_frames:
            messagebox.showwarning("هشدار", "ابتدا مدل سه‌بعدی را بازسازی کنید.")
            return
        if self.report_callback:
            self.report_callback(self.cine_frames[self.current_frame_idx])
            messagebox.showinfo("موفقیت", "تصویر نمای سه‌بعدی فعلی با موفقیت به مستندات گزارش بالینی بیمار افزوده شد.")
        else:
            messagebox.showinfo("موفقیت", "تصاویر ارتوگونال ۳D در پوشه پروژه ذخیره شدند و آماده استفاده در گزارش Word و PDF هستند.")

    def _save_cine_series(self):
        if not self.cine_frames:
            messagebox.showwarning("هشدار", "هنوز فریم سه‌بعدی تولید نشده است.")
            return
        d = filedialog.askdirectory(title="انتخاب مسیر ذخیره سری چرخشی ۳D")
        if d:
            for idx, f in enumerate(self.cine_frames):
                cv2.imwrite(os.path.join(d, f"Boali_3D_Cine_{idx+1:03d}.jpg"), f)
            messagebox.showinfo("ذخیره موفق", f"تمام {len(self.cine_frames)} فریم سه‌بعدی چرخشی با موفقیت در مسیر زیر ذخیره شدند:\n{d}")

    def _export_stl(self):
        if self.engine.volume_iso is None:
            messagebox.showwarning("هشدار", "ابتدا مقاطع سی‌تی اسکن را بارگذاری کنید.")
            return
        f = filedialog.asksaveasfilename(title="ذخیره فایل مش سه‌بعدی ارتوپدی", defaultextension=".stl",
                                         filetypes=[("STL 3D Mesh", "*.stl")])
        if f:
            try:
                v, fc = self.engine.export_stl_mesh(f, threshold=self.threshold_var.get())
                messagebox.showinfo("خروجی پرینتر ۳D", f"فایل مش سه‌بعدی با موفقیت استخراج شد:\nتعداد راس‌ها: {v}\nتعداد مثلث‌ها: {fc}\nمسیر: {f}")
            except Exception as e:
                messagebox.showerror("خطا", f"خطا در ایجاد فایل STL:\n{e}")
