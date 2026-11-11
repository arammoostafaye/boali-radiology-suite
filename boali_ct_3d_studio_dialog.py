"""
Boali Hospital Radiology AI Platform - 3D CT Reconstruction Studio Dialog
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
نسخه v3.0.2

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
        self.title("استودیوی بازسازی سه‌بعدی سی‌تی اسکن | نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (v3.0.2) [UI r2]")
        self.geometry("1180x780")
        self.minsize(1000, 680)
        self.configure(bg=C_CANVAS)

        self.engine = Boali3DReconstructor(target_resolution=(512, 512), output_size=(800, 800))
        self.default_folder = default_folder
        self.report_callback = report_callback

        self.loaded_slices_count = 0
        self.cine_frames = []
        self.current_frame_idx = 0
        self.is_processing = False

        self._build_ui()
        self._update_pacs_info()

        # Bind orbital rotation to mouse wheel across entire studio dialog
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", lambda e: self._on_mouse_wheel(e, delta=120))
        self.bind("<Button-5>", lambda e: self._on_mouse_wheel(e, delta=-120))

        if self.default_folder and os.path.exists(self.default_folder):
            self._detect_organ_and_setup_preset(self.default_folder)
            self._start_reconstruction_thread()
        elif getattr(self.master, "ct_slices", None) and len(self.master.ct_slices) >= 4:
            self._detect_organ_and_setup_preset()
            self._start_reconstruction_thread()

    def _detect_organ_and_setup_preset(self, folder_path=None):
        """بررسی هوشمند تگ‌های دایکام جهت انتخاب خودکار ارگان و آستانه هانسفیلد تشخیصی."""
        body_part = ""
        study_desc = ""
        series_desc = ""
        proto_name = ""

        # بررسی متادیتای مقاطع پکس در حافظه
        master_meta = getattr(self.master, "ct_slice_meta", None)
        if master_meta and len(master_meta) > 0:
            m0 = master_meta[0]
            body_part = str(m0.get("body_part", m0.get("BodyPartExamined", ""))).upper()
            study_desc = str(m0.get("study_description", m0.get("StudyDescription", ""))).upper()
            series_desc = str(m0.get("series_description", m0.get("SeriesDescription", ""))).upper()
            proto_name = str(m0.get("protocol_name", m0.get("ProtocolName", ""))).upper()

        if not (body_part or study_desc or series_desc) and folder_path and os.path.exists(folder_path):
            try:
                import pydicom
                for root, _, files in os.walk(folder_path):
                    for f in sorted(files):
                        fl = f.lower()
                        if fl in ["dicomdir", "thumbs.db", "desktop.ini"] or fl.endswith((".xml", ".txt", ".json", ".zip", ".html", ".exe", ".dll", ".log", ".bat")):
                            continue
                        fp = os.path.join(root, f)
                        if os.path.isfile(fp) and os.path.getsize(fp) > 512:
                            try:
                                ds = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
                                body_part = str(getattr(ds, "BodyPartExamined", "")).upper()
                                study_desc = str(getattr(ds, "StudyDescription", "")).upper()
                                series_desc = str(getattr(ds, "SeriesDescription", "")).upper()
                                proto_name = str(getattr(ds, "ProtocolName", "")).upper()
                                if body_part or study_desc or series_desc:
                                    break
                            except Exception:
                                continue
                    if body_part or study_desc or series_desc:
                        break
            except Exception:
                pass

        info_text = f"{body_part} {study_desc} {series_desc} {proto_name}".upper()

        if any(w in info_text for w in ["HEAD", "BRAIN", "SKULL", "SINUS", "FACIAL", "MANDIBLE", "ORBIT", "CALVARIA"]):
            matched_key = "CRANIOFACIAL"
        elif any(w in info_text for w in ["CHEST", "THORAX", "LUNG", "RIB", "STERN", "MEDIASTIN"]):
            matched_key = "CHEST_RIBCAGE"
        elif any(w in info_text for w in ["SPINE", "LUMBAR", "CERVICAL", "THORACIC", "VERTEBR", "L-S", "C-S", "DORSAL"]):
            matched_key = "SPINE_VERTEBRAE"
        elif any(w in info_text for w in ["PELVI", "ABDOM", "ABD", "HIP", "SACR", "ILIAC", "ACETABUL", "KUB"]):
            matched_key = "PELVIS_ACETABULUM"
        elif any(w in info_text for w in ["FOOT", "ANKLE", "TOE", "TARS", "CALCANE", "TALUS", "METATARS"]):
            matched_key = "FOOT_DORSAL"
        else:
            matched_key = "BONE_TRAUMA"

        if matched_key in PRESETS:
            preset = PRESETS[matched_key]
            rec_th = preset.get("hu_threshold", 95)
            self.preset_var.set(f"{matched_key} - {preset['title_fa']}")
            self.threshold_var.set(rec_th)
            if hasattr(self, "scale_th"):
                self.scale_th.set(rec_th)
            if hasattr(self, "lbl_th_val"):
                self.lbl_th_val.config(text=f"مقدار فعلی: {rec_th}")
            if hasattr(self, "lbl_desc"):
                self.lbl_desc.config(text=f"🎯 ارگان تشخیص داده‌شده: {preset['title_fa']}\n{preset['description']}")

    def _build_ui(self):
        # 1. Top Header
        top_bar = tk.Frame(self, bg=C_HEADER, height=54, highlightthickness=1, highlightbackground=C_BORDER)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        lbl_brand = tk.Label(
            top_bar,
            text="🏥 نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی | ماژول بازسازی سه‌بعدی سی‌تی اسکن (v3.0.2)",
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

        # Direct PACS Integration Card (Matching Mammography Workflow)
        pacs_box = tk.LabelFrame(left_panel, text="فراخوانی سری دایکام از پکس مدال (Medal PACS)",
                                 font=("Vazirmatn", 8, "bold"), fg=C_CYAN, bg=C_CARD, labelanchor="ne")
        pacs_box.pack(fill=tk.X, padx=12, pady=6)

        btn_pacs_ct = tk.Button(
            pacs_box,
            text="🌐 دریافت مستقیم مقاطع سی‌تی اسکن از پکس مدال",
            font=("Vazirmatn", 9, "bold"),
            bg="#0284c7",
            fg="white",
            activebackground="#0369a1",
            relief=tk.FLAT,
            pady=7,
            cursor="hand2",
            command=self._open_pacs_for_ct
        )
        btn_pacs_ct.pack(fill=tk.X, padx=8, pady=(6, 4))

        self.lbl_pacs_info = tk.Label(
            pacs_box,
            text="وضعیت دایکام: در انتظار دریافت از پکس مدال...",
            font=("Vazirmatn", 8),
            fg=C_TEXT_MUTED,
            bg=C_CARD,
            justify=tk.RIGHT,
            wraplength=310
        )
        self.lbl_pacs_info.pack(anchor=tk.E, padx=8, pady=(2, 6))

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
            "BONE_TRANSPARENT_SKIN - استخوان با بافت نرم شفاف (3D Ghosting)",
            "CRANIOFACIAL - جمجمه، فک و صورت",
            "SPINE_VERTEBRAE - ستون فقرات و مهره‌ها",
            "PELVIS_ACETABULUM - حلقه لگن و استابولوم",
            "VASCULAR_CTA_MIP - آنژیوگرافی عروق (3D MIP)"
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

        self.scale_th = ttk.Scale(th_box, from_=75, to_=220, variable=self.threshold_var, orient=tk.HORIZONTAL,
                                  command=lambda v: self.lbl_th_val.config(text=f"مقدار فعلی: {int(float(v))}"))
        self.scale_th.set(self.threshold_var.get())
        self.scale_th.pack(fill=tk.X, padx=8, pady=4)

        # Smoothness Slider (حذف پله‌ها و دنده‌ها و صیقل‌کاری کورتکس استخوان)
        sm_box = tk.LabelFrame(left_panel, text="میزان صیقل و رفع پله‌ها (3D Smoothing / mm)", font=("Vazirmatn", 8, "bold"),
                               fg=C_CYAN, bg=C_CARD, labelanchor="ne")
        sm_box.pack(fill=tk.X, padx=12, pady=6)

        self.smooth_var = tk.DoubleVar(value=0.60)
        self.lbl_sm_val = tk.Label(sm_box, text=f"مقدار فعلی: {self.smooth_var.get():.2f} mm (استاندارد کوثر/هیتاچی)", font=("Vazirmatn", 8),
                                   fg="#34d399", bg=C_CARD)
        self.lbl_sm_val.pack(anchor=tk.E, padx=8, pady=(2, 0))

        def _on_sm_change(v):
            val = float(v)
            if val < 0.20:
                hint = "حداقل (جزئیات تیز و ریزشکستگی‌ها)"
                color = "#fbbf24"
            elif val <= 0.65:
                hint = "استاندارد صیقلی (مشابه کوثر)"
                color = "#34d399"
            else:
                hint = "فوق‌العاده صاف و ابریشمی"
                color = "#38bdf8"
            self.lbl_sm_val.config(text=f"مقدار: {val:.2f} mm ({hint})", fg=color)
            if hasattr(self, "engine") and hasattr(self.engine, "volume_render16") and self.engine.volume_render16 is not None:
                # Live instant update button hint
                if hasattr(self, "btn_live_smooth"):
                    self.btn_live_smooth.config(state=tk.NORMAL)

        self.scale_sm = ttk.Scale(sm_box, from_=0.0, to_=1.20, variable=self.smooth_var, orient=tk.HORIZONTAL,
                                  command=_on_sm_change)
        self.scale_sm.set(0.60)
        self.scale_sm.pack(fill=tk.X, padx=8, pady=4)

        btn_sm_row = tk.Frame(sm_box, bg=C_CARD)
        btn_sm_row.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Button(btn_sm_row, text="تیز (0.35)", font=("Vazirmatn", 7), bg="#1e293b", fg=C_TEXT_LIGHT, relief=tk.FLAT,
                  command=lambda: [self.scale_sm.set(0.35), _on_sm_change(0.35)]).pack(side=tk.RIGHT, padx=2)
        tk.Button(btn_sm_row, text="استاندارد (0.60)", font=("Vazirmatn", 7), bg="#0f766e", fg="white", relief=tk.FLAT,
                  command=lambda: [self.scale_sm.set(0.60), _on_sm_change(0.60)]).pack(side=tk.RIGHT, padx=2)
        tk.Button(btn_sm_row, text="بسیار نرم (0.85)", font=("Vazirmatn", 7), bg="#1e293b", fg=C_TEXT_LIGHT, relief=tk.FLAT,
                  command=lambda: [self.scale_sm.set(0.85), _on_sm_change(0.85)]).pack(side=tk.RIGHT, padx=2)

        self.btn_live_smooth = tk.Button(
            sm_box,
            text="⚡ رندر لحظه‌ای با درجه صیقل جدید",
            font=("Vazirmatn", 8, "bold"),
            bg="#0284c7",
            fg="white",
            relief=tk.FLAT,
            command=self._apply_live_smooth_render
        )
        self.btn_live_smooth.pack(fill=tk.X, padx=8, pady=(2, 6))

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

        self.studio_view_mode = tk.StringVar(value="3d_vr")
        mode_box = tk.Frame(v_head, bg=C_HEADER)
        mode_box.pack(side=tk.RIGHT, padx=8)

        tk.Radiobutton(mode_box, text="🧊 بازسازی ۳D", variable=self.studio_view_mode, value="3d_vr",
                       command=self._on_view_mode_toggle, bg=C_HEADER, fg=C_CYAN, selectcolor=C_CANVAS,
                       activebackground=C_HEADER, activeforeground=C_CYAN, font=("Vazirmatn", 8, "bold")).pack(side=tk.RIGHT, padx=2)

        tk.Radiobutton(mode_box, text="🔬 مقاطع محوری ۲D", variable=self.studio_view_mode, value="2d_axial",
                       command=self._on_view_mode_toggle, bg=C_HEADER, fg=C_TEXT_LIGHT, selectcolor=C_CANVAS,
                       activebackground=C_HEADER, activeforeground=C_CYAN, font=("Vazirmatn", 8, "bold")).pack(side=tk.RIGHT, padx=2)

        lbl_hint = tk.Label(
            v_head,
            text="💡 برای چرخش ۳۶۰ درجه، اسکرول موس را روی تصویر بچرخانید",
            font=("Vazirmatn", 8),
            fg=C_TEXT_MUTED,
            bg=C_HEADER
        )
        lbl_hint.pack(side=tk.LEFT, padx=12, pady=6)

        # Canvas for 3D Image Display
        # قاب ثابت نگه‌دارنده: اندازه‌ی آن مستقل از تصویر داخلش است (pack_propagate=False).
        # اگر خود Label مبنای اندازه‌گیری باشد، هر فریم چند پیکسل بزرگ‌تر می‌شود (زوم تدریجی)
        # و نوار چرخش/دکمه‌های پایین از صفحه بیرون می‌روند.
        self.canvas_holder = tk.Frame(right_panel, bg=C_CANVAS)
        self.canvas_holder.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas_holder.pack_propagate(False)
        self._resize_job = None
        self.canvas_holder.bind("<Configure>", self._on_holder_resize)
        self.canvas_3d = tk.Label(self.canvas_holder, bg=C_CANVAS, bd=0, highlightthickness=0)
        self.canvas_3d.place(relx=0.5, rely=0.5, anchor="center")
        for w in (self.canvas_holder, self.canvas_3d):
            w.bind("<MouseWheel>", self._on_mouse_wheel)
            w.bind("<Button-1>", self._on_mouse_press)
            w.bind("<B1-Motion>", self._on_mouse_drag)
            w.bind("<Button-4>", lambda e: self._on_mouse_wheel(e, delta=120))   # Linux scroll up
            w.bind("<Button-5>", lambda e: self._on_mouse_wheel(e, delta=-120))  # Linux scroll down

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
        btn_report.pack(side=tk.RIGHT, padx=5)

        btn_viewer_gif = tk.Button(
            export_bar,
            text="🌐 استخراج ویوور تعاملی وب و GIF ۳۶۰°",
            font=("Vazirmatn", 8, "bold"),
            bg=C_CYAN,
            fg="#0b1120",
            relief=tk.FLAT,
            command=self._export_interactive_viewer_and_gif
        )
        btn_viewer_gif.pack(side=tk.RIGHT, padx=5)

        btn_cine_save = tk.Button(
            export_bar,
            text="💾 پکیج کامل فریم‌ها",
            font=("Vazirmatn", 8),
            bg=C_BORDER,
            fg=C_TEXT_LIGHT,
            relief=tk.FLAT,
            command=self._save_cine_series
        )
        btn_cine_save.pack(side=tk.RIGHT, padx=5)

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

    def _open_pacs_for_ct(self):
        try:
            from BoneSuppression_Desktop import MedalPacsModal
            parent_pacs = getattr(self.master, "pacs_client", None)
            modal = MedalPacsModal(self, pacs_client=parent_pacs)
            modal.target_modality.set("ct3d")
        except Exception as e:
            messagebox.showinfo("پکس مدال", f"خطا در باز کردن درگاه پکس مدال:\n{e}")

    def _update_pacs_info(self):
        master_slices = getattr(self.master, "ct_slices", None)
        master_folder = getattr(self.master, "ct_series_folder", "")
        folder = self.default_folder or master_folder
        patient_info = getattr(self.master, "patient_info", {})
        pname = patient_info.get("name") or "بیمار بالینی"
        pid = patient_info.get("id") or "---"

        if (folder and os.path.exists(folder)) or (master_slices and len(master_slices) >= 4):
            count = len(master_slices) if master_slices else 0
            if count == 0 and folder and os.path.exists(folder):
                try:
                    count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.getsize(os.path.join(folder, f)) > 512])
                except Exception:
                    count = 0
            self.lbl_pacs_info.config(
                text=f"✓ بیمار: {pname} | شناسه: {pid}\n✓ تعداد: {count} مقطع دایکام آماده بازسازی سه‌بعدی",
                fg="#34d399"
            )
            self.btn_run.config(state=tk.NORMAL, bg=C_BLUE)
            self._detect_organ_and_setup_preset(folder if (folder and os.path.exists(folder)) else None)
        else:
            self.lbl_pacs_info.config(
                text="⚠️ هنوز مقاطع دایکام از پکس دریافت نشده است.\nجهت شروع روی دکمه آبی دریافت مستقیم از پکس کلیک فرمایید.",
                fg=C_TEXT_MUTED
            )

    def _on_preset_change(self, event=None):
        val = self.preset_var.get()
        p_key = val.split(" - ")[0].strip()
        if p_key in PRESETS:
            preset = PRESETS[p_key]
            self.lbl_desc.config(text=preset["description"])
            new_th = preset.get("hu_threshold", 95)
            self.threshold_var.set(new_th)
            if hasattr(self, "scale_th"):
                self.scale_th.set(new_th)
            self.lbl_th_val.config(text=f"مقدار فعلی: {new_th}")

    def _start_reconstruction_thread(self):
        if self.is_processing:
            return
        master_slices = getattr(self.master, "ct_slices", None)
        master_folder = getattr(self.master, "ct_series_folder", "")
        folder = self.default_folder or master_folder

        has_slices = (master_slices and len(master_slices) >= 4)
        has_folder = (folder and os.path.exists(folder))

        if not has_slices and not has_folder:
            messagebox.showwarning(
                "عدم دریافت دایکام از پکس",
                "هنوز سری مقاطع دایکام سی‌تی اسکن از پکس مدال دریافت نشده است.\n\nلطفاً ابتدا روی دکمه «دریافت مستقیم مقاطع سی‌تی اسکن از پکس مدال» کلیک فرمایید."
            )
            return

        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, bg=C_BORDER)
        self.progress_bar["value"] = 5
        self.lbl_status.config(text="در حال آماده‌سازی و بازسازی سه‌بعدی...")

        import threading
        threading.Thread(target=self._run_reconstruction, daemon=True).start()

    def _run_reconstruction(self):
        try:
            val = self.preset_var.get()
            preset_key = val.split(" - ")[0].strip()
            num_frames = self.frames_var.get()
            th = self.threshold_var.get()
            if preset_key in PRESETS:
                PRESETS[preset_key]["hu_threshold"] = th

            # تنظیم درجه صیقل و رفع پله‌ها طبق انتخاب دستی کاربر
            sm_val = getattr(self, "smooth_var", None)
            if sm_val is not None:
                self.engine.render_smooth_mm = float(sm_val.get())

            def progress_cb(pct, msg):
                self.after(0, lambda: self._update_progress(pct, msg))

            # 1. Load slices & resample (prioritize direct DICOM files for genuine 16-bit HU data)
            master_slices = getattr(self.master, "ct_slices", None)
            master_folder = getattr(self.master, "ct_series_folder", "")
            folder = self.default_folder or master_folder

            if folder and os.path.exists(folder):
                progress_cb(12, "بارگذاری مستقیم مقاطع دایکام پکس با دانسیته واقعی هانسفیلد...")
                self.engine.load_series_from_folder(folder, slice_thickness=1.25, progress_callback=progress_cb)
            elif master_slices and len(master_slices) >= 4:
                progress_cb(12, f"استفاده از {len(master_slices)} مقطع آماده پکس در حافظه...")
                self.engine.load_series_from_slices(master_slices, slice_thickness=1.25, progress_callback=progress_cb)
            else:
                raise ValueError("پوشه یا مقاطع دایکام معتبری از پکس برای بازسازی یافت نشد.")

            # 2. Render 360 degree cine
            target_out = folder if (folder and os.path.exists(folder)) else os.path.abspath(os.path.dirname(__file__))
            out_cine_dir = os.path.join(target_out, "Boali_3D_Cine_Series")
            frame_paths = self.engine.generate_full_360_cine(out_cine_dir, num_frames=num_frames,
                                                            preset_key=preset_key, progress_callback=progress_cb)

            self.cine_frames = [cv2.imread(p) for p in frame_paths if os.path.exists(p)]

            self.after(0, self._on_reconstruction_done)
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e)
            self.after(0, lambda: self._on_reconstruction_error(err_msg))

    def _update_progress(self, pct, msg):
        self.progress_bar["value"] = pct
        self.lbl_status.config(text=msg)

    def _apply_live_smooth_render(self):
        """رندر آنی و لحظه‌ای فریم فعلی با مقدار هموارسازی و صیقل جدید بدون نیاز به بازسازی کل حجم"""
        if self.is_processing:
            return
        if not hasattr(self, "engine") or self.engine.volume_hu is None:
            messagebox.showinfo("راهنما", "لطفاً ابتدا سری دایکام را بارگذاری کرده و دکمه بازسازی را بزنید تا حجم سه‌بعدی تشکیل شود.")
            return

        sm_mm = float(self.smooth_var.get())
        self.engine.render_smooth_mm = sm_mm

        # اعمال فیلتر نرم روی حجم رندر در لحظه
        try:
            import scipy.ndimage as _ndi
            sp1 = self.engine.spacing
            if sm_mm > 0:
                sg = [sm_mm / max(0.1, s) for s in sp1]
                sm = _ndi.gaussian_filter(self.engine.volume_hu.astype(np.float32), sg)
                self.engine.volume_render16 = np.clip(sm + 1024.0, 0, 4095).astype(np.uint16)
            else:
                self.engine.volume_render16 = np.clip(self.engine.volume_hu.astype(np.float32) + 1024.0, 0, 4095).astype(np.uint16)

            # پاک کردن کش VTK تا با حجم صاف جدید مجدداً نگاشت شود
            self.engine._vtk_cache["preset_key"] = None

            # رندر تک‌فریم در زاویه جاری برای مشاهده آنی تغییرات توسط کاربر
            val = self.preset_var.get()
            preset_key = val.split(" - ")[0].strip()
            cur_angle = float(self.current_frame_idx * (360.0 / max(1, len(self.cine_frames)))) if self.cine_frames else 0.0
            new_frame = self.engine.render_shaded_view(angle_deg=cur_angle, preset_key=preset_key)

            if self.cine_frames and self.current_frame_idx < len(self.cine_frames):
                self.cine_frames[self.current_frame_idx] = new_frame
            self._display_frame(self.current_frame_idx)
            self.lbl_status.config(text=f"✓ درجه صیقل با موفقیت روی {sm_mm:.2f} mm تنظیم و تصویر در لحظه به‌روز شد.")
        except Exception as e:
            messagebox.showerror("خطا در رندر مجدد", f"خطا در اعمال صیقل لحظه‌ای:\n{e}")

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

    def _on_view_mode_toggle(self):
        if self.studio_view_mode.get() == "2d_axial":
            total_axial = self.engine.volume.shape[0] if (self.engine.volume is not None) else 0
            if total_axial > 0:
                self.slider_rot.config(from_=0, to=total_axial - 1)
                self.slider_rot.set(min(self.current_frame_idx, total_axial - 1))
                self._display_axial_slice(int(self.slider_rot.get()))
            else:
                messagebox.showinfo("اطلاع", "ابتدا مقاطع سی‌تی اسکن را بارگذاری فرمایید.")
        else:
            if self.cine_frames:
                self.slider_rot.config(from_=0, to=len(self.cine_frames) - 1)
                self.slider_rot.set(min(self.current_frame_idx, len(self.cine_frames) - 1))
                self._display_frame(int(self.slider_rot.get()))

    def _display_axial_slice(self, slice_idx):
        if self.engine.volume is None:
            return
        total = self.engine.volume.shape[0]
        slice_idx = max(0, min(slice_idx, total - 1))
        self.current_frame_idx = slice_idx
        self.lbl_angle.config(text=f"🔬 مقطع محوری ۲D سی‌تی: مقطع {slice_idx + 1} از {total}")

        sl = self.engine.volume[slice_idx].astype(np.uint8)
        sl_bgr = cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)

        canvas_h = self.canvas_holder.winfo_height() or 460
        canvas_w = self.canvas_holder.winfo_width() or 460
        h, w = sl_bgr.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 2.0)
        nw, nh = int(w * scale), int(h * scale)
        res = cv2.resize(sl_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)
        self.tk_photo = ImageTk.PhotoImage(im_pil)
        self.canvas_3d.config(image=self.tk_photo)

    def _display_frame(self, idx):
        if self.studio_view_mode.get() == "2d_axial":
            self._display_axial_slice(idx)
            return
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
        canvas_h = self.canvas_holder.winfo_height() or 460
        canvas_w = self.canvas_holder.winfo_width() or 460
        scale = min(canvas_w / w, canvas_h / h, 2.0)
        nw, nh = int(w * scale), int(h * scale)
        res = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)
        self.tk_photo = ImageTk.PhotoImage(im_pil)
        self.canvas_3d.config(image=self.tk_photo)

    def _on_holder_resize(self, event=None):
        """با تغییر اندازه‌ی پنجره، فریم فعلی دوباره متناسب با قاب نمایش داده می‌شود."""
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(80, lambda: self._display_frame(self.current_frame_idx))

    def _on_mouse_press(self, event):
        self._last_drag_x = event.x
        self._last_drag_y = event.y

    def _on_mouse_drag(self, event):
        if not hasattr(self, "_last_drag_x") or not self.cine_frames:
            return
        dx = event.x - self._last_drag_x
        if abs(dx) >= 8:
            step = 1 if dx > 0 else -1
            self._step_frame(-step)
            self._last_drag_x = event.x

    def _on_mouse_wheel(self, event, delta=None):
        d = delta if delta is not None else event.delta
        self._step_frame(-1 if d > 0 else 1)
        # جلوگیری از اجرای دوباره‌ی رویداد روی پنجره‌ی والد (قبلاً هر نوچ اسکرول دو فریم می‌پرید)
        return "break"

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

    def _export_interactive_viewer_and_gif(self):
        """استخراج همزمان ویوور تعاملی وب ۳۶۰ درجه و انیمیشن متحرک GIF با مقیاس کاملاً ثابت."""
        if not self.cine_frames:
            messagebox.showwarning("هشدار", "هنوز فریم سه‌بعدی بازسازی نشده است.")
            return

        out_dir = filedialog.askdirectory(title="انتخاب پوشه ذخیره ویوور تعاملی و GIF ۳۶۰ درجه")
        if not out_dir:
            return

        try:
            # 1. تولید انیمیشن متحرک گیف بدون زوم (دقیقاً مشابه خروجی استاندارد بوعلی)
            gif_path = os.path.join(out_dir, "Boali_3D_360_Rotation_HD.gif")
            from PIL import Image
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in self.cine_frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=75,
                loop=0,
                optimize=True
            )

            # 2. تولید فایل HTML ویوور تعاملی وب
            html_path = os.path.join(out_dir, "Boali_3D_360_Interactive_Viewer.html")
            import base64
            frames_b64 = []
            for f in self.cine_frames:
                _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf).decode("ascii")
                frames_b64.append(f"data:image/jpeg;base64,{b64}")

            html_content = self._build_html_viewer_template(frames_b64)
            with open(html_path, "w", encoding="utf-8") as hf:
                hf.write(html_content)

            ans = messagebox.askyesno(
                "استخراج موفقیت‌آمیز ویوور ۳۶۰ درجه",
                f"پکیج خروجی سه‌بعدی با موفقیت تولید شد:\n\n"
                f"• ویوور تعاملی تحت وب: {os.path.basename(html_path)}\n"
                f"• انیمیشن متحرک باکیفیت: {os.path.basename(gif_path)}\n\n"
                f"آیا مایلید ویوور تعاملی هم‌اکنون در مرورگر وب باز شود؟"
            )
            if ans:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
        except Exception as e:
            messagebox.showerror("خطا در استخراج ویوور ۳۶۰ درجه", str(e))

    def _build_html_viewer_template(self, frames_b64):
        """ایجاد کد کامل HTML5/CSS3/JS ویوور تعاملی ۳۶۰ درجه به صورت ۱۰۰٪ آفلاین و خودکفا."""
        import json
        frames_json = json.dumps(frames_b64)
        n_frames = len(frames_b64)
        pat_name = self.engine.metadata.get("PatientName", "بیمار بالینی") if hasattr(self.engine, "metadata") else "بیمار بالینی"
        study_desc = self.engine.metadata.get("StudyDescription", "CT Study") if hasattr(self.engine, "metadata") else "CT Study"

        return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ویوور تعاملی ۳۶۰ درجه سی‌تی اسکن بازسازی سه‌بعدی | بیمارستان بوعلی مریوان</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: #050811;
    color: #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow-x: hidden;
    user-select: none;
  }}
  .viewer-card {{
    background: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 14px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    width: 95%;
    max-width: 840px;
    margin: 20px auto;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .card-header {{
    background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #334155;
  }}
  .badge-hosp {{
    background: #0284c7;
    color: #fff;
    font-size: 11px;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 0.5px;
  }}
  .header-titles {{ text-align: right; }}
  .header-main {{ font-size: 14px; font-weight: 700; color: #f8fafc; }}
  .header-sub {{ font-size: 11px; color: #94a3b8; margin-top: 2px; direction: ltr; }}
  .viewport-wrapper {{
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    max-height: 640px;
    background: #000000;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: grab;
  }}
  .viewport-wrapper:active {{ cursor: grabbing; }}
  img#ctFrame {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
  }}
  .hud-top-left {{
    position: absolute;
    top: 14px;
    left: 16px;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 11px;
    color: #38bdf8;
    font-family: monospace;
    direction: ltr;
    pointer-events: none;
  }}
  .hud-top-right {{
    position: absolute;
    top: 14px;
    right: 16px;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 11px;
    color: #f59e0b;
    font-weight: 700;
    pointer-events: none;
  }}
  .hud-bottom-hint {{
    position: absolute;
    bottom: 12px;
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(4px);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 10px;
    color: #94a3b8;
    pointer-events: none;
  }}
  .controls-panel {{
    background: #0f172a;
    padding: 16px 20px;
    border-top: 1px solid #1e293b;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .slider-row {{ display: flex; align-items: center; gap: 12px; }}
  .angle-label {{
    font-family: monospace;
    font-size: 13px;
    font-weight: 700;
    color: #38bdf8;
    min-width: 75px;
    text-align: center;
    direction: ltr;
  }}
  .scrubber {{
    flex: 1;
    -webkit-appearance: none;
    height: 6px;
    border-radius: 3px;
    background: #334155;
    outline: none;
    cursor: pointer;
  }}
  .scrubber::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.7);
    cursor: pointer;
  }}
  .btn-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }}
  .btn-group-left {{ display: flex; gap: 6px; }}
  .btn-group-right {{ display: flex; gap: 6px; }}
  .ctrl-btn {{
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .ctrl-btn:hover {{
    background: #0284c7;
    color: #fff;
    border-color: #38bdf8;
  }}
  .ctrl-btn.active {{
    background: #0284c7;
    color: #fff;
    border-color: #38bdf8;
  }}
  .card-footer {{
    background: #0b1120;
    padding: 10px 20px;
    border-top: 1px solid #1e293b;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    color: #64748b;
  }}
</style>
</head>
<body>
<div class="viewer-card">
  <div class="card-header">
    <div class="badge-hosp">BOALI 3D CT STUDIO</div>
    <div class="header-titles">
      <div class="header-main">بازسازی تعاملی ۳۶۰ درجه سی‌تی اسکن (مدل استخوان)</div>
      <div class="header-sub">Patient: {pat_name} | {study_desc}</div>
    </div>
  </div>
  <div class="viewport-wrapper" id="viewport">
    <img id="ctFrame" src="{frames_b64[0]}" alt="3D CT Scan Frame">
    <div class="hud-top-left" id="hudAngle">0° | AP (Anterior)</div>
    <div class="hud-top-right">3D CT RECONSTRUCTION</div>
    <div class="hud-bottom-hint">🖱️ با درگ موس یا اسکرول چرخ موس بچرخانید</div>
  </div>
  <div class="controls-panel">
    <div class="slider-row">
      <button class="ctrl-btn active" id="btnPlay" onclick="togglePlay()">⏸ توقف</button>
      <input type="range" class="scrubber" id="angleSlider" min="0" max="{n_frames - 1}" value="0" oninput="onSliderChange(this.value)">
      <div class="angle-label" id="angleText">0° AP</div>
    </div>
    <div class="btn-row">
      <div class="btn-group-left">
        <button class="ctrl-btn active" id="btnAP" onclick="jumpToAngle(0)">روبرو (0° AP)</button>
        <button class="ctrl-btn" id="btnObl" onclick="jumpToAngle({n_frames // 4})">مایل (90° Lat)</button>
        <button class="ctrl-btn" id="btnPA" onclick="jumpToAngle({n_frames // 2})">پشت (180° PA)</button>
        <button class="ctrl-btn" id="btnMed" onclick="jumpToAngle({(3 * n_frames) // 4})">داخلی (270° Med)</button>
      </div>
      <div class="btn-group-right">
        <button class="ctrl-btn" onclick="stepFrame(-1)">◀ قبل</button>
        <button class="ctrl-btn" onclick="stepFrame(1)">بعد ▶</button>
      </div>
    </div>
  </div>
  <div class="card-footer">
    <div>مرکز آموزشی درمانی بیمارستان بوعلی مریوان - آرام مصطفائی | v3.0.2</div>
    <div style="direction: ltr; font-family: monospace;">800×800 Invariant Scale | Pure Skeleton</div>
  </div>
</div>
<script>
  const FRAMES = {frames_json};
  const TOTAL_FRAMES = FRAMES.length;
  let currentIdx = 0;
  let isPlaying = true;
  let playInterval = null;
  let imgEl = document.getElementById("ctFrame");
  let sliderEl = document.getElementById("angleSlider");
  let angleTextEl = document.getElementById("angleText");
  let hudAngleEl = document.getElementById("hudAngle");
  let btnPlayEl = document.getElementById("btnPlay");
  let viewportEl = document.getElementById("viewport");

  function getAngleDesc(idx) {{
    let deg = Math.round((idx / TOTAL_FRAMES) * 360);
    let desc = "";
    if (deg >= 345 || deg <= 15) desc = "AP (قدامی)";
    else if (deg >= 75 && deg <= 105) desc = "Lateral (طرفی)";
    else if (deg >= 165 && deg <= 195) desc = "PA (خلفی)";
    else if (deg >= 255 && deg <= 285) desc = "Medial (داخلی)";
    else if (deg < 90) desc = "Antero-Lateral";
    else if (deg < 180) desc = "Postero-Lateral";
    else if (deg < 270) desc = "Postero-Medial";
    else desc = "Antero-Medial";
    return {{ deg, desc }};
  }}

  function setFrame(idx) {{
    currentIdx = (idx + TOTAL_FRAMES) % TOTAL_FRAMES;
    imgEl.src = FRAMES[currentIdx];
    sliderEl.value = currentIdx;
    let {{ deg, desc }} = getAngleDesc(currentIdx);
    angleTextEl.textContent = deg + "° " + desc.split(" ")[0];
    hudAngleEl.textContent = deg + "° | " + desc;
    document.querySelectorAll(".btn-group-left .ctrl-btn").forEach(b => b.classList.remove("active"));
    if (currentIdx === 0) document.getElementById("btnAP")?.classList.add("active");
    else if (currentIdx === Math.floor(TOTAL_FRAMES/4)) document.getElementById("btnObl")?.classList.add("active");
    else if (currentIdx === Math.floor(TOTAL_FRAMES/2)) document.getElementById("btnPA")?.classList.add("active");
    else if (currentIdx === Math.floor((3*TOTAL_FRAMES)/4)) document.getElementById("btnMed")?.classList.add("active");
  }}

  function onSliderChange(val) {{
    if (isPlaying) stopPlay();
    setFrame(parseInt(val, 10));
  }}

  function stepFrame(delta) {{
    if (isPlaying) stopPlay();
    setFrame(currentIdx + delta);
  }}

  function jumpToAngle(frameIdx) {{
    if (isPlaying) stopPlay();
    setFrame(frameIdx);
  }}

  function togglePlay() {{
    if (isPlaying) stopPlay();
    else startPlay();
  }}

  function startPlay() {{
    isPlaying = true;
    btnPlayEl.textContent = "⏸ توقف";
    btnPlayEl.classList.add("active");
    playInterval = setInterval(() => {{ setFrame(currentIdx + 1); }}, 75);
  }}

  function stopPlay() {{
    isPlaying = false;
    btnPlayEl.textContent = "▶ پخش ۳۶۰°";
    btnPlayEl.classList.remove("active");
    if (playInterval) {{ clearInterval(playInterval); playInterval = null; }}
  }}

  let isDragging = false, startX = 0;
  viewportEl.addEventListener("mousedown", e => {{ isDragging = true; startX = e.clientX; if (isPlaying) stopPlay(); }});
  window.addEventListener("mousemove", e => {{
    if (!isDragging) return;
    let dx = e.clientX - startX;
    if (Math.abs(dx) >= 12) {{
      let steps = Math.floor(dx / 12);
      setFrame(currentIdx - steps);
      startX = e.clientX;
    }}
  }});
  window.addEventListener("mouseup", () => {{ isDragging = false; }});
  viewportEl.addEventListener("wheel", e => {{
    e.preventDefault();
    if (isPlaying) stopPlay();
    setFrame(currentIdx + (e.deltaY < 0 ? -1 : 1));
  }});

  setFrame(0);
  startPlay();
</script>
</body>
</html>
"""

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
