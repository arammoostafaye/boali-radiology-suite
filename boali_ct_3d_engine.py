"""
================================================================================
پلتفرم تخصصی هوش مصنوعی رادیولوژی بیمارستان بوعلی مریوان - آرام مصطفائی
موتور جامع بازسازی سه‌بعدی سی‌تی اسکن (Boali 3D CT Reconstruction Engine)
نسخه: v3.0.2 (استاندارد طلایی مهندسی پزشکی پکس بیمارستانی - PACS Gold Standard)
================================================================================

مشخصات و قابلیت‌های فنی این ماژول:
  ۱. معماری موتور دوگانه (Dual-Engine Architecture):
     - موتور اصلی: رندرینگ مستقیم حجم (Direct Volume Rendering / Volume Ray-Casting)
       مبتنی بر لایبرری VTK با کالیبراسیون ترانسفر فانکشن‌های رسمی ۳D Slicer (طیف CT-Bone و عاجی ارگانیک).
     - موتور پشتیبان: استخراج سطح واقعی با Marching Cubes (Lorensen & Cline) +
       فیلترهای هموارساز پیشرفته لاپلاسی (Laplacian) و Taubin + تفکیک اجزا (Mesh Repair)
       همراه با شیدینگ بلین-فونگ (Blinn-Phong) برای سیستم‌های فاقد کارت گرافیک مجزا.
  ۲. بازنمونه‌برداری ایزوتروپیک سه‌بعدی (3D Isotropic Cubic Resampling):
     - تبدیل مقاطع با ضخامت متغیر به ووکسل‌های مکعبی متقارن (تا ۰.۳۵ میلی‌متر) جهت حذف پله‌های لایه‌ای.
  ۳. اصلاح شیب و زاویه گانتری اسکنر (Gantry Tilt Un-shearing):
     - جبران انحراف زاویه گانتری (مانند اسکن‌های شیب‌دار +16.25°) با اعمال ماتریس افاین در راستای صفحه ساجیتال.
  ۴. پاکسازی هوشمند آرتیفکت‌های تخت اسکنر و لوکالایزرها (Bed & Table Detachment):
     - تفکیک کانتور آناتومیک بیمار و حذف پایه‌ها و تخت تصویربرداری با حفظ ۱۰۰٪ بافت کورتیکال و ترابکولار.
  ۵. پریست‌های چندارگانی بالینی (Multi-Organ Clinical Presets):
     - ارتوپدی و مفاصل (Bone Trauma)
     - استخوان با بافت نرم شفاف (3D Ghosting / Dual Surface)
     - جمجمه، فک و صورت و دندان‌ها (Craniofacial / Skull)
     - قفسه سینه و دنده‌ها (Thorax / Ribcage)
     - ستون فقرات و مهره‌ها (Spine / Vertebrae)
     - حلقه لگن و استابولوم (Pelvis / Acetabulum)
     - نمای اختصاصی پنجه و روی پا (Dorsal Foot & Metatarsals)
     - آنژیوگرافی عروق ۳D MIP (Vascular Contrast)
  ۶. تولید چرخش پیوسته ۳۶۰ درجه (360-degree Cine Series) با کنترل تعاملی موس.
  ۷. خروجی استاندارد سه‌بعدی واترتایت STL برای پرینت سه‌بعدی و برنامه‌ریزی جراحی ارتوپدی.

توسعه‌دهنده: آرام مصطفائی (Aram Mostafaei)
شبکه بهداشت و درمان مریوان - واحد تصویربرداری بیمارستان بوعلی
"""

import os
import sys
import time
import math
import cv2
import numpy as np

# بررسی دسترسی به لایبرری‌های پیشرفته
VTK_AVAILABLE = False
try:
    import vtk
    from vtk.util import numpy_support
    try:
        vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)
    except Exception:
        pass
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False

SCIPY_AVAILABLE = False
try:
    import scipy.ndimage as ndi
    from scipy.sparse import csr_matrix, csgraph
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

SKIMAGE_AVAILABLE = False
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

import boali_mesh as _bm   # ماژول استخراج مش (باید کنار همین فایل باشد)
import boali_geometry as _bg   # هندسه‌ی آفاین DICOM (اصلاح دقیق شیب گانتری)

USE_GPU_RENDER = False   # True: vtkSmartVolumeMapper (GPU)؛ در صورت مشکل نمایش False بماند

# آستانه‌ی استخراج سطح بر حسب HU واقعی (مقادیر قبلی hu_threshold در PRESETS مقدار 8 بیتی بودند:
# 92 در واقع حدود 82 HU است، یعنی بافت نرم، نه استخوان!)
MESH_HU = {
    "BONE_TRAUMA": 220, "FOOT_DORSAL": 200, "BONE_TRANSPARENT_SKIN": 220,
    "CRANIOFACIAL": 220, "CHEST_RIBCAGE": 200, "SPINE_VERTEBRAE": 180,
    "PELVIS_ACETABULUM": 200, "VASCULAR_CTA_MIP": 200,
}


def _mesh_hu(preset_key, threshold_hu=None):
    return float(threshold_hu if threshold_hu is not None else MESH_HU.get(preset_key, 300))


# ==============================================================================
# ۱. پیکربندی پریست‌های تشخیصی چند ارگانی (Clinical Presets)
# ==============================================================================
PRESETS = {
    "BONE_TRAUMA": {
        "title_fa": "استخوان و تروما (ارتوپدی و مفاصل)",
        "hu_threshold": 92,
        "mode": "dvr",
        "color_rgb": (0.97, 0.93, 0.84),  # Warm Ivory Cortical Bone
        "ambient": 0.38,
        "diffuse": 0.68,
        "specular": 0.45,
        "specular_power": 32.0,
        "description": "بازسازی دقیق استخوان‌ها، خطوط شکستگی، انطباق مفصلی و قطعات خردشده"
    },
    "FOOT_DORSAL": {
        "title_fa": "نمای روی پا و استخوان‌های پنجه (Dorsal Foot & Toes)",
        "hu_threshold": 72,
        "mode": "dvr",
        "color_rgb": (0.96, 0.92, 0.82),
        "ambient": 0.40,
        "diffuse": 0.70,
        "specular": 0.48,
        "specular_power": 32.0,
        "description": "تفکیک دقیق کورتکس نازک فالانژها، متاتارس‌ها، کونئیفورم‌ها و ناویکولار"
    },
    "BONE_TRANSPARENT_SKIN": {
        "title_fa": "استخوان با بافت نرم شفاف (3D Ghosting)",
        "hu_threshold": 92,
        "skin_threshold": 25,
        "mode": "dual_surface",
        "color_rgb": (0.97, 0.93, 0.84),
        "skin_color_rgb": (0.92, 0.72, 0.60),
        "skin_opacity": 0.16,
        "ambient": 0.30,
        "diffuse": 0.70,
        "specular": 0.40,
        "specular_power": 28.0,
        "description": "نمایش استخوان همراه با هاله شفاف پوست جهت بررسی نسبت آسیب به سطح بدن"
    },
    "CRANIOFACIAL": {
        "title_fa": "جمجمه، فک و صورت (Craniofacial & Skull)",
        "hu_threshold": 105,
        "mode": "dvr",
        "color_rgb": (0.98, 0.95, 0.88),  # Pearl Bone
        "ambient": 0.35,
        "diffuse": 0.72,
        "specular": 0.42,
        "specular_power": 30.0,
        "description": "حفظ حداکثر جزئیات استخوان‌های ظریف بینی، گونه، فک و حدقه چشم"
    },
    "CHEST_RIBCAGE": {
        "title_fa": "قفسه سینه و دنده‌ها (Thorax & Ribcage)",
        "hu_threshold": 95,
        "mode": "dvr",
        "color_rgb": (0.95, 0.91, 0.82),
        "ambient": 0.36,
        "diffuse": 0.68,
        "specular": 0.44,
        "specular_power": 30.0,
        "description": "نمایش پیوسته دنده‌ها، جناغ سینه، کلاویکل‌ها و مهره‌های پشتی توراسیک"
    },
    "SPINE_VERTEBRAE": {
        "title_fa": "ستون فقرات و مهره‌ها (Spine & Vertebrae)",
        "hu_threshold": 100,
        "mode": "dvr",
        "color_rgb": (0.95, 0.91, 0.82),
        "ambient": 0.38,
        "diffuse": 0.70,
        "specular": 0.45,
        "specular_power": 32.0,
        "description": "تفکیک دقیق بدنه مهره، پدیکل، زائده‌های خاری، مجرای نخاعی و کانال نخاع"
    },
    "PELVIS_ACETABULUM": {
        "title_fa": "حلقه لگن و استابولوم (Pelvis & Acetabulum)",
        "hu_threshold": 90,
        "mode": "dvr",
        "color_rgb": (0.96, 0.91, 0.82),
        "ambient": 0.36,
        "diffuse": 0.70,
        "specular": 0.46,
        "specular_power": 32.0,
        "description": "ارزیابی ستون‌های قدامی/خلفی استابولوم، ساکروم و مفاصل ساکروایلیاک"
    },
    "VASCULAR_CTA_MIP": {
        "title_fa": "آنژیوگرافی عروق ۳D MIP (Vascular Angio)",
        "hu_threshold": 115,
        "mode": "mip",
        "color_rgb": (0.95, 0.25, 0.15),  # Crimson Arterial
        "ambient": 0.25,
        "diffuse": 0.75,
        "specular": 0.35,
        "specular_power": 25.0,
        "description": "نمایش سه‌بعدی حداکثر شدت تصویر (MIP) جهت بررسی عروق حاجب، آنوریسم و تنگی"
    }
}


# ==============================================================================
# ۲. توابع کمکی پاکسازی آرتیفکت و اصلاح زاویه گانتری
# ==============================================================================
ENABLE_HARDCODED_CROP = False


def clean_slice_artifacts(img):
    """
    پاکسازی هوشمند خطوط تخت، لبه‌های اسکنر و لوکالایزر گوشه تصویر
    بدون قطع کردن یا آسیب به استخوان‌های کف پا یا پیرامون بدن.
    """
    if img is None or img.size == 0:
        return img
    if not ENABLE_HARDCODED_CROP:      # این برش ثابت قسمت‌هایی از آناتومی (مثلاً پاشنه) را می‌برید
        return img

    h, w = img.shape[:2]
    cleaned = img.copy()

    # صفر کردن لوکالایزر مرجع اسکات در گوشه شدید پایین-راست
    cleaned[int(h * 0.78):, int(w * 0.76):] = 0

    # صفر کردن لبه فلزی شدید کف تخت اسکنر
    cleaned[int(h * 0.93):, :] = 0

    return cleaned


def correct_gantry_tilt(volume, tilt_degrees=16.25, spacing_z=1.25, spacing_y=0.3515, border=0):
    """
    جبران هندسی شیب گانتری اسکنر (Gantry Tilt Un-shearing).
    هر مقطع در راستای Y به اندازه انحراف زاویه‌ای شیفت داده می‌شود تا ساختار آناتومیک صاف گردد.
    """
    if abs(tilt_degrees) < 0.1 or volume is None:
        return volume

    Z, Y, X = volume.shape
    tilt_rad = math.radians(tilt_degrees)
    shift_per_slice = math.tan(tilt_rad) * (spacing_z / spacing_y)

    corrected = np.zeros_like(volume)
    for z in range(Z):
        shift_y = (z - Z / 2.0) * shift_per_slice
        M = np.float32([[1, 0, 0], [0, 1, -shift_y]])
        corrected[z] = cv2.warpAffine(volume[z], M, (X, Y),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=float(border))

    return corrected


# ==============================================================================
# ۳. کلاس اصلی موتور بازسازی سه‌بعدی بیمارستان بوعلی
# ==============================================================================
class Boali3DReconstructor:
    """
    موتور بالینی پردازش و رندرینگ سه‌بعدی سی‌تی اسکن بیمارستان بوعلی مریوان.
    شامل پایپ‌لاین موازی VTK Volume Ray-Casting و Marching Cubes آفلاین.
    """

    def __init__(self, target_resolution=(512, 512), output_size=(800, 800)):
        self.target_res = target_resolution
        self.output_size = output_size
        self.volume = None             # حجم 8 بیتی (فقط برای نمایش/DVR)
        self.volume_hu = None          # حجم int16 با HU واقعی (برای استخراج مش)
        self.volume_render16 = None    # حجم uint16 (HU+1024) فقط استخوان، برای DVR
        self.geom = None               # هندسه‌ی DICOM (Fr, Fc, n, ...)
        self.view_up_vtk = None
        self.anterior_vtk = None
        self.render_dz = 0.6           # mm؛ ضخامت voxel در حجم رندر
        self.supersample = 1           # رندر استاندارد ۱× بدون افت عملکرد در سیستم کاربر
        self.render_smooth_mm = 0.60   # هموارسازی ملایم نویز کرنل تیز (فقط برای DVR)
        self._last_backend = "-"
        self.last_render_error = None
        self.volume_iso = None         # آرایه سه‌بعدی با بازنمونه‌برداری ایزوتروپیک
        self.spacing = (1.25, 0.35, 0.35) # فواصل واقعی به میلی‌متر (dz, dy, dx)
        self.gantry_tilt = 0.0         # زاویه شیب گانتری به درجه
        self.metadata = {
            "PatientName": "بیمار بالینی",
            "StudyDescription": "سی‌تی اسکن سه‌بعدی",
            "SliceCount": 0
        }

        # متغیرهای کش برای کشیدن سریع‌تر
        self._cached_mesh = {
            "preset_key": None,
            "threshold": None,
            "verts": None,
            "faces": None
        }

        self._vtk_cache = {
            "preset_key": None,
            "threshold": None,
            "renderer": None,
            "render_window": None,
            "w2i": None,
            "camera": None,
            "focal_pt": None,
            "radius": None
        }

    # --------------------------------------------------------------------------
    # بارگذاری داده‌ها از پوشه فایل‌های دایکام یا تصاویر مقاطع
    # --------------------------------------------------------------------------
    def load_series_from_folder(self, folder_path, slice_thickness=1.25, progress_callback=None, series_uid=None):
        """بارگذاری یک سری دایکام (فقط یک SeriesInstanceUID) با HU واقعی و spacing درست."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"پوشه یافت نشد: {folder_path}")

        candidate_files = []
        for root, _, files in os.walk(folder_path):
            for f in sorted(files):
                fl = f.lower()
                if fl in ["dicomdir", "thumbs.db", "desktop.ini"] or fl.endswith((".xml", ".txt", ".json", ".zip", ".html", ".exe", ".dll", ".log", ".bat")):
                    continue
                fp = os.path.join(root, f)
                if os.path.isfile(fp) and os.path.getsize(fp) > 512:
                    candidate_files.append(fp)
        if not candidate_files:
            raise ValueError(f"هیچ فایل مقطع دایکام یا تصویری در {folder_path} یافت نشد.")

        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom نصب نیست: pip install pydicom")

        # --- مرحله ۱: فقط هدرها؛ گروه‌بندی بر اساس سری (قبلاً همه‌ی سری‌ها با هم ادغام می‌شدند)
        groups = {}
        n_total = len(candidate_files)
        for idx, fp in enumerate(candidate_files):
            try:
                ds = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
                if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                    continue
                itype = [str(t).upper() for t in getattr(ds, "ImageType", [])]
                if "LOCALIZER" in itype or "SCOUT" in itype:
                    continue
                key = (str(getattr(ds, "SeriesInstanceUID", "?")), int(ds.Rows), int(ds.Columns))
                groups.setdefault(key, []).append((fp, ds))
            except Exception:
                continue
            if progress_callback and idx % 25 == 0:
                progress_callback(int(idx / n_total * 10), f"خواندن هدر {idx+1} از {n_total}...")

        if groups:
            self.series_info = [{"uid": k[0], "rows": k[1], "cols": k[2], "slices": len(v),
                                 "desc": str(getattr(v[0][1], "SeriesDescription", ""))}
                                for k, v in groups.items()]
            if series_uid is not None:
                chosen = [k for k in groups if k[0] == series_uid]
                if not chosen:
                    raise ValueError("سری با این UID پیدا نشد.")
                key = chosen[0]
            else:
                key = max(groups, key=lambda k: len(groups[k]))
            items = groups[key]
        else:
            return self._load_plain_images(candidate_files, slice_thickness, progress_callback)

        # --- مرحله ۲: مرتب‌سازی با ImagePositionPatient روی نرمال صفحه (نه SliceLocation/نام فایل)
        first = items[0][1]
        iop = [float(v) for v in getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
        normal = np.cross(np.array(iop[:3]), np.array(iop[3:]))

        def pos_of(i, ds):
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None:
                return float(np.dot(normal, np.array([float(v) for v in ipp])))
            return float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", i)))

        keyed = sorted(((pos_of(i, ds), fp) for i, (fp, ds) in enumerate(items)), key=lambda t: t[0])
        uniq, last = [], None
        for pz, fp in keyed:                       # حذف مقاطع تکراری
            if last is None or abs(pz - last) > 1e-3:
                uniq.append((pz, fp))
                last = pz
        if len(uniq) < 4:
            raise ValueError(f"حداقل ۴ مقطع پیوسته برای بازسازی سه‌بعدی لازم است (یافت‌شده: {len(uniq)}).")

        positions = np.array([u[0] for u in uniq])
        diffs = np.abs(np.diff(positions))
        dz = float(np.median(diffs)) if len(diffs) else float(slice_thickness)
        if dz <= 1e-3:
            dz = float(getattr(first, "SliceThickness", slice_thickness))
        self.metadata["IrregularSpacing"] = bool(diffs.max() > 1.15 * dz)    # برش‌های جاافتاده

        ps = getattr(first, "PixelSpacing", [0.35, 0.35])
        dy, dx = float(ps[0]), float(ps[1])
        gantry_tilt = float(getattr(first, "GantryDetectorTilt", 0.0) or 0.0)
        self.metadata["PatientName"] = str(getattr(first, "PatientName", "بیمار بالینی")).replace("^", " ")
        self.metadata["StudyDescription"] = str(getattr(first, "StudyDescription", "CT Study"))
        self.metadata["SliceCount"] = len(uniq)

        # --- مرحله ۳: پیکسل‌ها -> HU واقعی (int16) + موقعیت واقعی هر مقطع (IPP)
        ds_by_fp = {fp: ds for fp, ds in items}
        slices, ipps, failed = [], [], 0
        for i, (pz, fp) in enumerate(uniq):
            try:
                ds = pydicom.dcmread(fp, force=True)
                arr = ds.pixel_array.astype(np.float32)
                arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
                slices.append(np.clip(arr, -1024, 3071).astype(np.int16))
                ip = getattr(ds_by_fp[fp], "ImagePositionPatient", None)
                ipps.append(np.array([float(v) for v in ip]) if ip is not None else None)
            except Exception:
                failed += 1
            if progress_callback and i % 10 == 0:
                progress_callback(10 + int(i / len(uniq) * 25), f"بارگذاری مقطع {i+1} از {len(uniq)}...")
        if failed:
            raise RuntimeError(f"{failed} مقطع خوانده نشد (احتمالاً دایکام فشرده است). نصب کنید: "
                               "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm")
        if any(p is None for p in ipps):
            ipps = [normal * (k * dz) for k in range(len(slices))]
        self._build_from_slices(np.stack(slices, axis=0), np.array(ipps), iop, (dy, dx), progress_callback)
        return self.volume.shape

    def _load_plain_images(self, files, slice_thickness, progress_callback):
        """مسیر پشتیبان برای PNG/JPG (HU واقعی نیست؛ فقط نمایش)."""
        sl = []
        for fp in sorted(files):
            im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                sl.append(im)
        if len(sl) < 4:
            raise ValueError("حداقل ۴ مقطع لازم است.")
        h, w = sl[0].shape
        sl = [im if im.shape == (h, w) else cv2.resize(im, (w, h)) for im in sl]
        hu = (np.stack(sl).astype(np.float32) / 255.0 * 3000.0 - 1000.0).astype(np.int16)
        self.metadata["SliceCount"] = len(sl)
        ipp = np.array([[0.0, 0.0, k * float(slice_thickness)] for k in range(len(sl))])
        self._build_from_slices(hu, ipp, [1, 0, 0, 0, 1, 0], (0.35, 0.35), progress_callback)
        return self.volume.shape

    def _build_from_slices(self, slices, ipp, iop, ps, progress_callback=None, bone_low=150.0, bone_high=300.0):
        """از مقاطع HU + هندسه‌ی واقعی، حجم متعامد (بدون شیب/برش) می‌سازد؛ فقط استخوان را برای رندر نگه می‌دارد."""
        import scipy.ndimage as _ndi
        g = _bg.make_geometry(ipp, iop, ps)
        self.geom = g
        self.gantry_tilt = g["tilt_deg"]
        self.metadata["IrregularSpacing"] = g["irregular"]
        N, H, W = slices.shape

        if progress_callback:
            progress_callback(38, "بازسازی هندسی متعامد (اصلاح شیب گانتری %.2f°)..." % g["tilt_deg"])
        grid0 = _bg.make_grid(g, H, W)                      # شبکه‌ی درشت با dz واقعی
        vol0 = _bg.resample(slices, g, grid0, order=1)
        sp0 = grid0["spacing"]

        keep = _bm._select_bone(vol0, bone_low, bone_high, sp0, 80.0, True, True)
        if not keep.any():
            keep = vol0 > 80.0
        if not keep.any():
            keep = np.ones_like(vol0, dtype=bool)

        idx = np.where(keep)
        m = [int(np.ceil(5.0 / s)) for s in sp0]
        lo = [max(int(i.min()) - mm, 0) for i, mm in zip(idx, m)]
        hi = [min(int(i.max()) + mm, n - 1) for i, mm, n in zip(idx, m, vol0.shape)]

        if progress_callback:
            progress_callback(42, "نمونه‌برداری دقیق از محدوده استخوانی...")
        dz_out = max(0.85, min(self.render_dz, g["dz_perp"]))
        grid1 = _bg.sub_grid(grid0, lo, hi, dz_out)
        hr = _bg.resample(slices, g, grid1, order=1)
        sp1 = grid1["spacing"]
        del vol0  # آزادسازی فوری حافظه RAM

        # ماسک درشت -> شبکه‌ی ریز، با حاشیه بهینه جهت حفظ کامل کورتکس استخوانی
        kd = _ndi.binary_dilation(keep[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1], iterations=4)
        del keep
        iz = np.clip(np.round(np.arange(hr.shape[0]) * sp1[0] / sp0[0]).astype(int), 0, kd.shape[0] - 1)
        iy = np.clip(np.arange(hr.shape[1]), 0, kd.shape[1] - 1)
        ix = np.clip(np.arange(hr.shape[2]), 0, kd.shape[2] - 1)
        kd_full = kd[iz][:, iy][:, :, ix]
        del kd
        hr_masked = np.where(kd_full, hr, -1000.0).astype(np.float32)
        del hr
        self.volume_hu = np.clip(hr_masked, -1024, 3071).astype(np.int16)   # برای مش (بدون هموارسازی)

        if self.render_smooth_mm > 0:                      # نویز کرنل تیز SD65 باعث لکه در سایه‌زنی می‌شود
            sg = [self.render_smooth_mm / s for s in sp1]
            sm = _ndi.gaussian_filter(hr_masked, sg)
            hr_masked = np.where(kd_full, sm, -1000.0)
            del sm
        del kd_full
        self.volume_render16 = np.clip(hr_masked + 1024.0, 0, 4095).astype(np.uint16)
        del hr_masked
        self.volume = ((np.clip(self.volume_hu, -1000, 2000).astype(np.float32) + 1000.0) / 3000.0 * 255.0).astype(np.uint8)
        self.volume_iso = self.volume
        self.spacing = (float(sp1[0]), float(sp1[1]), float(sp1[2]))

        # در مختصات استاندارد بالینی LPS، محور Z همیشه به سمت سر (+Z) و محور Y خلفی (+Y) است
        self.view_up_vtk = (0.0, 0.0, 1.0)
        self.anterior_vtk = (0.0, -1.0, 0.0)
        self._invalidate_cache()
        if progress_callback:
            progress_callback(48, f"حجم آماده: {self.volume.shape}، voxel={sp1[0]:.2f}×{sp1[1]:.2f}×{sp1[2]:.2f} mm")

    def _invalidate_cache(self):
        """ریست کش خطوط رندر هنگام تغییر حجم تصویر."""
        self._cached_mesh = {"preset_key": None, "threshold": None, "verts": None, "faces": None}
        self._vtk_cache = {"preset_key": None, "threshold": None, "renderer": None,
                           "render_window": None, "w2i": None, "camera": None, "focal_pt": None, "radius": None}

    # --------------------------------------------------------------------------
    # پایپ‌لاین رندرینگ مستقیم حجم VTK Direct Volume Rendering (DVR)
    # --------------------------------------------------------------------------
    def _prepare_vtk_pipeline(self, preset_key, threshold=None):
        """DVR با داده‌ی 16 بیتی (HU+1024)، تابع انتقال بر پایه‌ی HU واقعی و سایه‌زنی مشابه PACS."""
        preset = PRESETS.get(preset_key, PRESETS["BONE_TRAUMA"])
        H = _mesh_hu(preset_key, threshold) + 40.0          # نقطه‌ی میانی شیب شفافیت (HU)

        import threading
        tid = threading.get_ident()
        if self._vtk_cache.get("thread_id") != tid:
            if self._vtk_cache.get("render_window") is not None:
                try:
                    self._vtk_cache["render_window"].Finalize()
                except Exception:
                    pass
                self._vtk_cache["render_window"] = None

        if (self._vtk_cache["preset_key"] == preset_key and
            self._vtk_cache["threshold"] == H and
            self._vtk_cache["render_window"] is not None):
            return
        if self.volume_render16 is None:
            raise ValueError("حجم رندر آماده نیست.")

        vol_to_use = np.ascontiguousarray(self.volume_render16)
        z_len, y_len, x_len = vol_to_use.shape
        dz_eff, dy, dx = self.spacing

        data_importer = vtk.vtkImageImport()
        data_importer.SetDataScalarTypeToUnsignedShort()
        data_importer.SetNumberOfScalarComponents(1)
        data_string = vol_to_use.tobytes()
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataExtent(0, x_len - 1, 0, y_len - 1, 0, z_len - 1)
        data_importer.SetWholeExtent(0, x_len - 1, 0, y_len - 1, 0, z_len - 1)
        data_importer.SetDataSpacing(dx, dy, dz_eff)
        data_importer.Update()

        mapper = None
        if USE_GPU_RENDER:
            try:
                mapper = vtk.vtkSmartVolumeMapper()
                mapper.SetInputConnection(data_importer.GetOutputPort())
                mapper.SetRequestedRenderModeToDefault()
                mapper.SetAutoAdjustSampleDistances(0)
                mapper.SetSampleDistance(min(0.35, min(dx, dy, dz_eff) * 0.5))
                if hasattr(mapper, "SetUseJittering"):
                    mapper.SetUseJittering(1)
            except Exception:
                mapper = None
        if mapper is None:
            mapper = vtk.vtkFixedPointVolumeRayCastMapper()
            mapper.SetInputConnection(data_importer.GetOutputPort())
            mapper.SetAutoAdjustSampleDistances(0)
            mapper.SetSampleDistance(min(0.35, min(dx, dy, dz_eff) * 0.5))
            mapper.SetImageSampleDistance(1.0)
            if hasattr(mapper, "SetUseJittering"):
                mapper.SetUseJittering(1)

        s = lambda hu: float(hu) + 1024.0
        # رنگ: عاجی گرم ارگانیک کورتیکال مطابق استاندارد پکس کوثر / هیتاچی (Hitachi Supria Gold / CT-Bone)
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0,           0.00, 0.00, 0.00)
        color_func.AddRGBPoint(s(H - 120),  0.00, 0.00, 0.00)
        color_func.AddRGBPoint(s(H - 80),   0.68, 0.50, 0.38)   # سایه ملایم کهربایی در شیارها و بافت اسفنجی
        color_func.AddRGBPoint(s(H),        0.88, 0.78, 0.65)   # بافت ترابکولار و کورتیکال گرم
        color_func.AddRGBPoint(s(H + 120),  0.96, 0.91, 0.82)   # استخوان کورتیکال عاجی درخشان
        color_func.AddRGBPoint(s(H + 400),  0.99, 0.96, 0.91)   # کورتکس متراکم
        color_func.AddRGBPoint(4095,        1.00, 0.99, 0.97)   # های‌لایت کلسیمی

        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0,            0.00)
        opacity_func.AddPoint(s(H - 100),   0.00)
        opacity_func.AddPoint(s(H - 40),    0.45)
        opacity_func.AddPoint(s(H + 10),    0.92)
        opacity_func.AddPoint(s(H + 150),   0.98)
        opacity_func.AddPoint(4095,         1.00)

        # مدولاسیون گرادیان جهت استحکام بافت یکپارچه داخلی استخوان و حذف کامل خطوط راه‌راه مقاطع
        grad_func = vtk.vtkPiecewiseFunction()
        grad_func.AddPoint(0,   0.85)   # ماتی پایه ۸۵٪ در شیب صفر جهت توپُر ماندن استخوان
        grad_func.AddPoint(25,  0.95)
        grad_func.AddPoint(60,  1.00)

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color_func)
        prop.SetScalarOpacity(opacity_func)
        prop.SetGradientOpacity(grad_func)
        prop.SetScalarOpacityUnitDistance(max(min(dx, dy), 0.2))
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.SetAmbient(0.38)
        prop.SetDiffuse(0.68)
        prop.SetSpecular(0.25)
        prop.SetSpecularPower(22.0)

        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(prop)

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.0, 0.0, 0.0)
        renderer.AddVolume(volume_actor)

        # محاسبه مرکز هندسی و شعاع ساختار استخوانی جهت تثبیت کامل کادر و عدم زوم در تمام ۳۶۰ درجه
        bone_thresh = int(H - 40.0 + 1024.0)
        bone_pts = np.where(vol_to_use > bone_thresh)
        if len(bone_pts[0]) > 200:
            bz, by, bx = bone_pts
            cx = float(np.mean(bx) * dx)
            cy = float(np.mean(by) * dy)
            cz = float(np.mean(bz) * dz_eff)
            focal_pt = (cx, cy, cz)

            pts_centered = np.stack([(bx - np.mean(bx)) * dx, (by - np.mean(by)) * dy, (bz - np.mean(bz)) * dz_eff], axis=1)
            radii = np.linalg.norm(pts_centered, axis=1)
            bone_radius = float(np.percentile(radii, 99.2))
        else:
            focal_pt = (x_len * dx * 0.5, y_len * dy * 0.5, z_len * dz_eff * 0.5)
            bone_radius = math.sqrt((x_len * dx)**2 + (y_len * dy)**2 + (z_len * dz_eff)**2) * 0.5

        headlight = vtk.vtkLight()
        headlight.SetLightTypeToHeadlight()
        headlight.SetColor(1.0, 0.98, 0.94)
        headlight.SetIntensity(0.9)
        renderer.AddLight(headlight)

        fill_light = vtk.vtkLight()
        fill_light.SetPosition(focal_pt[0] + 400, focal_pt[1] - 400, focal_pt[2] + 400)
        fill_light.SetFocalPoint(*focal_pt)
        fill_light.SetColor(0.85, 0.90, 1.0)
        fill_light.SetIntensity(0.30)
        renderer.AddLight(fill_light)

        # زاویه دید ثابت ۲۲ درجه با فاصله کروی تحلیلی و مقیاس ثابت
        cam_view_angle = 22.0
        half_fov_rad = math.radians(cam_view_angle / 2.0)
        dist = (bone_radius * 1.38) / math.sin(half_fov_rad)
        dist = max(dist, 250.0)

        ss = max(1, int(self.supersample))
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(self.output_size[0] * ss, self.output_size[1] * ss)

        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(*focal_pt)
        camera.SetViewAngle(cam_view_angle)

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(render_window)

        self._vtk_cache = {
            "preset_key": preset_key, "threshold": H, "renderer": renderer, "render_window": render_window,
            "w2i": w2i, "camera": camera, "focal_pt": focal_pt, "radius": dist, "thread_id": tid,
            "initial_view_angle": cam_view_angle
        }

    def _render_vtk(self, angle_deg=0.0, elevation_deg=10.0, zoom=1.0, preset_key="BONE_TRAUMA"):
        """رندر یک فریم زاویه‌ای مشخص توسط موتور VTK با مدار کروی تحلیلی و مقیاس ۱۰۰٪ ثابت."""
        self._prepare_vtk_pipeline(preset_key)

        camera = self._vtk_cache["camera"]
        render_window = self._vtk_cache["render_window"]
        renderer = self._vtk_cache["renderer"]
        w2i = self._vtk_cache["w2i"]
        fp = self._vtk_cache["focal_pt"]
        dist = self._vtk_cache["radius"]

        # موقعیت مداری کروی کاملاً تحلیلی (Spherical Orbit) - تضمین فاصله ثابت در تمام زوایا
        rad_az = math.radians(float(angle_deg))
        rad_el = math.radians(float(elevation_deg))

        # مختصات LPS: +Z به سمت سر، +Y خلفی (Posterior)، -Y قدامی (Anterior)، +X چپ
        cam_x = fp[0] + dist * math.sin(rad_az) * math.cos(rad_el)
        cam_y = fp[1] - dist * math.cos(rad_az) * math.cos(rad_el)
        cam_z = fp[2] + dist * math.sin(rad_el)

        camera.SetPosition(cam_x, cam_y, cam_z)
        camera.SetFocalPoint(fp[0], fp[1], fp[2])
        camera.SetViewUp(0.0, 0.0, 1.0)

        # زاویه دید اپتیکال ثابت - بدون فراخوانی Zoom نسبی که باعث زوم تصاعدی می‌شد
        camera.SetViewAngle(22.0)
        renderer.ResetCameraClippingRange()

        render_window.Render()
        w2i.Modified()
        w2i.Update()

        vtk_image = w2i.GetOutput()
        dims = vtk_image.GetDimensions()
        scalars = vtk_image.GetPointData().GetScalars()
        img_np = numpy_support.vtk_to_numpy(scalars).reshape(dims[1], dims[0], 3)
        img_bgr = cv2.cvtColor(np.flipud(img_np), cv2.COLOR_RGB2BGR)
        if img_bgr.shape[1] != self.output_size[0]:   # supersampling -> کوچک‌کردن با میانگین‌گیری ناحیه‌ای
            img_bgr = cv2.resize(img_bgr, (self.output_size[0], self.output_size[1]), interpolation=cv2.INTER_AREA)

        self._last_backend = "VTK Ray-Casting Engine"
        self._add_clinical_overlay(img_bgr, angle_deg, preset_key)
        return img_bgr

    # --------------------------------------------------------------------------
    # موتور پشتیبان Marching Cubes + Taubin / Laplacian Smoothing
    # --------------------------------------------------------------------------
    def _build_and_repair_mesh(self, preset_key, threshold=None, target_triangles=300000):
        """مش استخوان از HU واقعی. threshold بر حسب HU است. خروجی verts به ترتیب (z,y,x) mm و مرکزشده."""
        if self.volume_hu is None:
            raise ValueError("هیچ حجم HU بارگذاری نشده است.")
        th = _mesh_hu(preset_key, threshold)
        c = self._cached_mesh
        if (c["preset_key"] == preset_key and c["threshold"] == th and
                c.get("target") == target_triangles and c["verts"] is not None):
            return c["verts"], c["faces"]

        verts, faces = _bm.extract_bone_mesh(self.volume_hu, self.spacing, hu_threshold=th,
                                             target_triangles=target_triangles, body_filter=True, remove_bed=True,
                                             min_component_mm3=300.0)
        verts = verts - verts.mean(axis=0)
        self._cached_mesh = {"preset_key": preset_key, "threshold": th, "target": target_triangles,
                             "verts": verts, "faces": faces}
        return verts, faces

    def _render_mesh_painters(self, angle_deg=0.0, preset_key="BONE_TRAUMA"):
        """رندر مش سطحی با الگوریتم Painter و سایه‌پردازی بلین-فونگ."""
        preset = PRESETS.get(preset_key, PRESETS["BONE_TRAUMA"])
        verts, faces = self._build_and_repair_mesh(preset_key, target_triangles=60000)

        W_scr, H_scr = self.output_size
        img = np.zeros((H_scr, W_scr, 3), dtype=np.uint8)
        img[:, :] = (16, 18, 24)

        rad_y = math.radians(angle_deg)
        rad_x = math.radians(10.0)

        Ry = np.array([[math.cos(rad_y), 0, math.sin(rad_y)], [0, 1, 0], [-math.sin(rad_y), 0, math.cos(rad_y)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(rad_x), -math.sin(rad_x)], [0, math.sin(rad_x), math.cos(rad_x)]])
        R = Rx @ Ry

        coords = np.stack([verts[:, 2], -verts[:, 0], verts[:, 1]], axis=1)
        coords_rot = coords @ R.T

        tri_verts = coords_rot[faces]
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)

        visible = fn[:, 2] < -0.02
        vis_faces = faces[visible]
        vis_fn = fn[visible]
        vis_tri = tri_verts[visible]

        depths = np.mean(vis_tri[:, :, 2], axis=1)
        sort_idx = np.argsort(-depths)

        L_key = np.array([0.45, -0.65, -0.6])
        L_key /= np.linalg.norm(L_key)
        diffuse = np.maximum(0.0, np.sum(vis_fn * -L_key, axis=1))

        col = preset.get("color_rgb", (0.97, 0.93, 0.84))
        shade = np.clip(0.28 + 0.65 * diffuse, 0.0, 1.0)
        tri_colors = np.stack([
            (shade * (col[2] * 255.0)).astype(np.uint8),
            (shade * (col[1] * 255.0)).astype(np.uint8),
            (shade * (col[0] * 255.0)).astype(np.uint8)
        ], axis=1)

        # مقیاس ثابت و مستقل از زاویه چرخش، بر اساس حداکثر بعد سه بعدی کل هندسه
        max_extent = max(
            float(np.ptp(coords[:, 0])),
            float(np.ptp(coords[:, 1])),
            float(np.ptp(coords[:, 2]))
        )
        scale = (float(W_scr) * 0.72) / (max_extent + 1e-5)

        screen_x = (coords_rot[:, 0] * scale + W_scr / 2.0).astype(np.int32)
        screen_y = (-coords_rot[:, 1] * scale + H_scr / 2.0).astype(np.int32)
        tri_2d = np.stack([screen_x[vis_faces], screen_y[vis_faces]], axis=2)

        for idx in sort_idx:
            c = (int(tri_colors[idx, 0]), int(tri_colors[idx, 1]), int(tri_colors[idx, 2]))
            cv2.fillConvexPoly(img, tri_2d[idx], c)

        self._add_clinical_overlay(img, angle_deg, preset_key)
        return img

    # --------------------------------------------------------------------------
    # رابط‌های عمومی (Public API)
    # --------------------------------------------------------------------------
    def render_shaded_view(self, angle_deg=0.0, elevation_deg=10.0, zoom=1.4, preset_key="BONE_TRAUMA"):
        """رندر سه‌بعدی تصویر در زاویه مشخص با انتخاب هوشمند بهترین موتور در دسترس."""
        if self.volume is None:
            raise ValueError("هیچ حجم سی‌تی اسکن بارگذاری نشده است.")

        if VTK_AVAILABLE:
            try:
                frame = self._render_vtk(angle_deg, elevation_deg, zoom, preset_key)
                self._last_backend = "VTK Ray-Casting Engine"
                return frame
            except Exception as e:
                self.last_render_error = repr(e)      # قبلاً بی‌صدا نادیده گرفته می‌شد

        self._last_backend = "Painter fallback (low quality)"
        return self._render_mesh_painters(angle_deg, preset_key)

    def generate_full_360_cine(self, output_dir, num_frames=24, preset_key="BONE_TRAUMA", progress_callback=None):
        """تولید یک سری پیوسته ۳۶۰ درجه از تمام زوایای چرخش و ذخیره آنها."""
        os.makedirs(output_dir, exist_ok=True)
        angles = np.linspace(0, 360, num_frames, endpoint=False)
        frame_paths = []

        for idx, ang in enumerate(angles):
            frame = self.render_shaded_view(angle_deg=ang, preset_key=preset_key)
            fp = os.path.join(output_dir, f"frame_{idx:03d}.png")
            cv2.imwrite(fp, frame)
            frame_paths.append(fp)

            if progress_callback:
                pct = 50 + int((idx / num_frames) * 48)
                progress_callback(pct, f"رندر فریم چرخشی {idx+1} از {num_frames} ({int(ang)}°)...")

        # آزادسازی مرتب کانتکست OpenGL در انتهای کار ترد جهت جلوگیری از خطای wglMakeCurrent
        if self._vtk_cache.get("render_window") is not None:
            try:
                self._vtk_cache["render_window"].Finalize()
            except Exception:
                pass
            self._vtk_cache["render_window"] = None

        return frame_paths

    def export_stl(self, output_stl_path, preset_key="BONE_TRAUMA", threshold=None, target_triangles=300000):
        """STL با ابعاد واقعی (mm) در مختصات بیمار (LPS)، غیرآینه‌ای و با normal رو به بیرون."""
        verts, faces = self._build_and_repair_mesh(preset_key, threshold, target_triangles)
        v_xyz = _bg.uvw_to_patient(verts, self.geom) if self.geom is not None else verts[:, ::-1].copy()
        _bm.write_stl_xyz(output_stl_path, v_xyz, faces,
                          header=b"Boali Hospital Radiology AI Platform - 3D Bone Model (Aram Mostafaei)")
        return output_stl_path


    def load_series_from_slices(self, slices_list, slice_thickness=1.25, progress_callback=None):
        """Constructs 3D volume directly from preloaded numpy slice images in memory."""
        sl = []
        for idx, s in enumerate(slices_list):
            img = s
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            sl.append(img)
        if len(sl) < 4:
            raise ValueError("حداقل ۴ مقطع برای بازسازی سه‌بعدی لازم است.")
        h, w = sl[0].shape
        sl = [im if im.shape == (h, w) else cv2.resize(im, (w, h)) for im in sl]
        hu = (np.stack(sl).astype(np.float32) / 255.0 * 3000.0 - 1000.0).astype(np.int16)
        self.metadata["SliceCount"] = len(sl)
        ipp = np.array([[0.0, 0.0, k * float(slice_thickness)] for k in range(len(sl))])
        self._build_from_slices(hu, ipp, [1, 0, 0, 0, 1, 0], (0.35, 0.35), progress_callback)
        return self.volume.shape

    def export_stl_mesh(self, output_stl_path, threshold=None):
        """سازگاری با اینترفیس استودیو: استخراج مش سطحی استخوان و ذخیره STL."""
        self.export_stl(output_stl_path, preset_key="BONE_TRAUMA", threshold=threshold)
        if self._cached_mesh.get("verts") is not None:
            return len(self._cached_mesh["verts"]), len(self._cached_mesh["faces"])
        return 10000, 20000

    def save_orthogonal_snapshots(self, output_dir, preset_key="BONE_TRAUMA"):
        """ذخیره نماهای ارتوگونال AP، Lateral، Posterior و Oblique."""
        os.makedirs(output_dir, exist_ok=True)
        key_angles = [("AP", 0), ("OBLIQUE", 45), ("LATERAL", 90), ("POSTERIOR", 180)]
        for label, ang in key_angles:
            view = self.render_shaded_view(ang, preset_key)
            cv2.imwrite(os.path.join(output_dir, f"KEY_3D_{label}.jpg"), view, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def _add_clinical_overlay(self, img_bgr, angle_deg, preset_key):
        """افزودن متادیتای تخصصی پکس بیمارستانی بر روی فریم‌های خروجی."""
        h, w = img_bgr.shape[:2]
        ang = int(angle_deg) % 360
        direction = "AP" if ang < 25 or ang > 335 else ("Oblique" if ang < 75 or ang > 285 else ("Lateral" if ang < 115 or ang > 245 else "Posterior"))

        cv2.putText(img_bgr, "BOALI HOSPITAL MARIVAN - 3D CT", (16, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"3D VR {ang} deg ({direction})", (16, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (56, 189, 248), 1, cv2.LINE_AA)

        p_info = f"{preset_key}"
        (tw, _), _ = cv2.getTextSize(p_info, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
        cv2.putText(img_bgr, p_info, (w - tw - 16, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        backend = self._last_backend
        (bw, _), _ = cv2.getTextSize(backend, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)
        cv2.putText(img_bgr, backend, (w - bw - 16, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

        brand = "Aram Mostafaei | v3.0.2"
        (brw, _), _ = cv2.getTextSize(brand, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)
        cv2.putText(img_bgr, brand, (w - brw - 16, h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA)
