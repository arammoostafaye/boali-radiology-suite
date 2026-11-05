"""
Boali Hospital Radiology AI Platform - 3D CT Reconstruction Studio Engine
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
نسخه v3.0.0

High-Speed Pure OpenCV/NumPy 3D Volume Rendering & Multi-Organ Surface Reconstruction Engine.
Requires ZERO external heavy scientific packages (No scipy, no external daemon required).
100% Offline, Native Windows Compatible.
Author: آرام مصطفائی (Aram Mostafaei)
Network: شبکه بهداشت و درمان مریوان - بیمارستان بوعلی
"""

import os
import sys
import time
import math
import numpy as np
import cv2

# Anatomical Presets Configuration
PRESETS = {
    "BONE_TRAUMA": {
        "title_fa": "استخوان و تروما (ارتوپدی و مفاصل)",
        "hu_threshold": 92,
        "upper_threshold": 1000,
        "smooth_sigma": 0.8,
        "palette": "WARM_IVORY",
        "description": "بهینه‌شده برای بررسی شکستگی‌ها، کندگی رباط‌ها و انطباق سطوح مفصلی"
    },
    "CRANIOFACIAL": {
        "title_fa": "جمجمه و فک و صورت (Craniofacial)",
        "hu_threshold": 110,
        "upper_threshold": 1200,
        "smooth_sigma": 0.6,
        "palette": "PEARL_BONE",
        "description": "حفظ جزئیات استخوان‌های نازک بینی، کاسه چشم و فک"
    },
    "SPINE_VERTEBRAE": {
        "title_fa": "ستون فقرات و مهره‌ها (Spine)",
        "hu_threshold": 100,
        "upper_threshold": 1100,
        "smooth_sigma": 0.75,
        "palette": "WARM_IVORY",
        "description": "نمایش بدنه مهره‌ها، پدیکل، لامینا و انحرافات ستون فقرات"
    },
    "PELVIS_ACETABULUM": {
        "title_fa": "حلقه لگن و استابولوم (Pelvis)",
        "hu_threshold": 95,
        "upper_threshold": 1000,
        "smooth_sigma": 0.9,
        "palette": "COPPER_BONE",
        "description": "بررسی ستون قدامی و خلفی استابولوم و مفاصل ساکروایلیاک"
    },
    "VASCULAR_CTA_MIP": {
        "title_fa": "آنژیوگرافی عروق (3D MIP Vascular)",
        "hu_threshold": 115,
        "upper_threshold": 450,
        "smooth_sigma": 0.5,
        "palette": "ANGIO_RED",
        "description": "نمایش حداکثر چگالی عروق خونی حاجب، آنوریسم‌ها و تنگی‌ها"
    },
    "AIRWAY_BRONCHO": {
        "title_fa": "راه‌های هوایی و نای (Virtual Airway)",
        "hu_threshold": 30,
        "upper_threshold": 75,
        "smooth_sigma": 1.0,
        "palette": "AIRWAY_CYAN",
        "description": "بازسازی فضای هوای داخل نای و مجاری تنفسی"
    }
}


def get_color_lut(palette_name):
    """Builds medical color lookup tables for realistic shaded volume rendering."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    if palette_name == "WARM_IVORY":
        # Warm golden ivory (Siemens syngo.via / GE AW bone look)
        for i in range(256):
            lut[i, 0, 0] = int(i * 0.70) # B
            lut[i, 0, 1] = int(i * 0.88) # G
            lut[i, 0, 2] = int(i * 1.00) # R
    elif palette_name == "COPPER_BONE":
        for i in range(256):
            lut[i, 0, 0] = int(i * 0.55)
            lut[i, 0, 1] = int(i * 0.75)
            lut[i, 0, 2] = int(i * 1.00)
    elif palette_name == "PEARL_BONE":
        for i in range(256):
            lut[i, 0, 0] = int(i * 0.82)
            lut[i, 0, 1] = int(i * 0.92)
            lut[i, 0, 2] = int(i * 0.98)
    elif palette_name == "ANGIO_RED":
        for i in range(256):
            lut[i, 0, 0] = int(i * 0.20)
            lut[i, 0, 1] = int(i * 0.45)
            lut[i, 0, 2] = int(i * 1.00)
    elif palette_name == "AIRWAY_CYAN":
        for i in range(256):
            lut[i, 0, 0] = int(i * 1.00)
            lut[i, 0, 1] = int(i * 0.85)
            lut[i, 0, 2] = int(i * 0.30)
    else:
        for i in range(256):
            lut[i, 0, 0] = i
            lut[i, 0, 1] = i
            lut[i, 0, 2] = i
    return lut


class Boali3DReconstructor:
    """Clinical 3D CT Reconstruction Engine for Boali Hospital."""

    def __init__(self, target_resolution=(256, 256)):
        self.target_res = target_resolution
        self.volume = None
        self.volume_iso = None
        self.slice_thickness = 1.25
        self.pixel_spacing = 0.5

    def load_series_from_folder(self, folder_path, slice_thickness=1.25, progress_callback=None):
        """Loads a stack of CT image files (.dcm or image formats) recursively and constructs an isotropic 3D volume."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Recursively search all subdirectories
        all_candidate_files = []
        for root, _, files in os.walk(folder_path):
            for f in sorted(files):
                fl = f.lower()
                if fl in ["dicomdir", "thumbs.db", "desktop.ini"] or fl.endswith((".xml", ".txt", ".json", ".zip", ".html", ".exe", ".dll", ".log", ".bat")):
                    continue
                fp = os.path.join(root, f)
                if os.path.isfile(fp) and os.path.getsize(fp) > 512:
                    all_candidate_files.append(fp)

        if not all_candidate_files:
            raise ValueError(f"هیچ مقطع تصویری یا دایکام در پوشه {folder_path} یافت نشد.")

        self.slice_thickness = slice_thickness
        total_files = len(all_candidate_files)
        slices_with_order = []

        for idx, file_path in enumerate(all_candidate_files):
            img = None
            sort_key = idx
            try:
                import pydicom
                ds = pydicom.dcmread(file_path, force=True)
                if hasattr(ds, 'pixel_array'):
                    raw = ds.pixel_array.astype(np.float32)
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        raw = raw * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                        raw = np.clip(raw, -1000, 1000)
                        img = ((raw + 1000.0) / 2000.0 * 255.0).astype(np.uint8)
                    else:
                        img = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-8) * 255.0).astype(np.uint8)
                    sort_key = float(getattr(ds, 'SliceLocation', getattr(ds, 'InstanceNumber', idx)))
            except Exception:
                pass

            if img is None and cv2 is not None:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                if (img.shape[1], img.shape[0]) != self.target_res:
                    img = cv2.resize(img, self.target_res, interpolation=cv2.INTER_AREA)

                # Clean outer boundary scanner lines
                img[:10, :] = 0
                img[-12:, :] = 0
                img[:, :10] = 0
                img[:, -12:] = 0
                # Zero out scout localizer in bottom-right corner if present
                h, w = img.shape
                img[int(h*0.82):, int(w*0.82):] = 0

                slices_with_order.append((sort_key, img))

            if progress_callback and idx % 10 == 0:
                progress_callback(int((idx / total_files) * 30), f"بارگذاری مقطع {idx+1} از {total_files}...")

        if len(slices_with_order) < 5:
            raise ValueError(f"حداقل ۵ مقطع پیوسته سی‌تی اسکن برای بازسازی سه‌بعدی مورد نیاز است (تعداد یافت‌شده: {len(slices_with_order)}).")

        slices_with_order.sort(key=lambda item: item[0])
        slices = [item[1] for item in slices_with_order]

        self.volume = np.stack(slices, axis=0).astype(np.float32)

        # Pure OpenCV/NumPy Isotropic Resampling along Z-axis
        z_dim = self.volume.shape[0]
        z_ratio = max(1.0, (self.slice_thickness / self.pixel_spacing))
        target_z = min(int(z_dim * z_ratio), 180)

        if progress_callback:
            progress_callback(40, "درون‌یابی ایزوتروپیک سه‌بعدی وکسل‌ها (Isotropic Resampling)...")

        vol_t = np.transpose(self.volume, (1, 0, 2))
        vol_resized_t = np.empty((self.volume.shape[1], target_z, self.volume.shape[2]), dtype=np.float32)
        for y in range(self.volume.shape[1]):
            vol_resized_t[y] = cv2.resize(vol_t[y], (self.volume.shape[2], target_z), interpolation=cv2.INTER_LINEAR)
        self.volume_iso = np.transpose(vol_resized_t, (1, 0, 2))

        # Smart Table / Scanner Bed Removal
        self._remove_scanner_couch()

        if progress_callback:
            progress_callback(50, "حذف خودکار تخت معاینه و تفکیک بافت‌ها...")

        return self.volume_iso.shape

    def load_series_from_slices(self, slices_list, slice_thickness=1.25, progress_callback=None):
        """Builds isotropic 3D volume directly from preloaded numpy slices (instant zero-disk lag)."""
        self.slice_thickness = slice_thickness
        processed = []
        for idx, s in enumerate(slices_list):
            img = s
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            if (img.shape[1], img.shape[0]) != self.target_res:
                img = cv2.resize(img, self.target_res, interpolation=cv2.INTER_AREA)
            img[:10, :] = 0
            img[-12:, :] = 0
            img[:, :10] = 0
            img[:, -12:] = 0
            h, w = img.shape
            img[int(h*0.82):, int(w*0.82):] = 0
            processed.append(img)

        self.volume = np.stack(processed, axis=0).astype(np.float32)

        z_dim = self.volume.shape[0]
        z_ratio = max(1.0, (self.slice_thickness / self.pixel_spacing))
        target_z = min(int(z_dim * z_ratio), 180)

        if progress_callback:
            progress_callback(40, "درون‌یابی ایزوتروپیک سه‌بعدی وکسل‌ها (Isotropic Resampling)...")

        vol_t = np.transpose(self.volume, (1, 0, 2))
        vol_resized_t = np.empty((self.volume.shape[1], target_z, self.volume.shape[2]), dtype=np.float32)
        for y in range(self.volume.shape[1]):
            vol_resized_t[y] = cv2.resize(vol_t[y], (self.volume.shape[2], target_z), interpolation=cv2.INTER_LINEAR)
        self.volume_iso = np.transpose(vol_resized_t, (1, 0, 2))
        self._remove_scanner_couch()

        if progress_callback:
            progress_callback(50, "حذف خودکار تخت معاینه و تفکیک بافت‌ها...")

        return self.volume_iso.shape

    def _remove_scanner_couch(self):
        """Detaches and zeroes out flat scanner cradle/bed pixels at bottom of volume."""
        if self.volume_iso is None:
            return
        z_dim, y_dim, x_dim = self.volume_iso.shape
        table_region_y = int(y_dim * 0.86)
        mask_table = (self.volume_iso[:, table_region_y:, :] > 100)
        col_sums = np.sum(mask_table, axis=1) # (Z, X)
        flat_table_line = col_sums > 6
        for z in range(z_dim):
            for x in range(x_dim):
                if flat_table_line[z, x]:
                    self.volume_iso[z, table_region_y:, x] = 0.0

    def render_shaded_view(self, angle_deg=0.0, preset_key="BONE_TRAUMA", light_dir=None):
        """Renders one high-quality shaded 3D perspective at given rotation angle using fast OpenCV routines."""
        if self.volume_iso is None:
            raise ValueError("No volume loaded. Call load_series_from_folder first.")

        preset = PRESETS.get(preset_key, PRESETS["BONE_TRAUMA"])
        threshold = preset["hu_threshold"]

        # Fast Gaussian blur on each slice using OpenCV
        vol_smooth = np.empty_like(self.volume_iso)
        for z in range(self.volume_iso.shape[0]):
            vol_smooth[z] = cv2.GaussianBlur(self.volume_iso[z], (5, 5), 0.8)

        # Fast 3D rotation around Z-axis using cv2.warpAffine
        if angle_deg != 0.0:
            h, w = self.volume_iso.shape[1], self.volume_iso.shape[2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rot = np.empty_like(vol_smooth)
            for z in range(vol_smooth.shape[0]):
                rot[z] = cv2.warpAffine(vol_smooth[z], M, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            rot = vol_smooth

        if light_dir is None:
            L = np.array([0.4, 0.8, -0.4])
        else:
            L = np.array(light_dir)
        L = L / np.linalg.norm(L)

        # Surface ray-casting along ray axis 1 (anterior-posterior depth)
        mask = rot > threshold
        hit = np.any(mask, axis=1)
        if not np.any(hit):
            return np.zeros((rot.shape[0], rot.shape[2], 3), dtype=np.uint8)

        depth = np.argmax(mask, axis=1).astype(np.float32)
        depth = cv2.GaussianBlur(depth, (5, 5), 1.0)

        # Surface normal from 2D depth gradients
        dz, dx = np.gradient(depth)
        nx = -dx
        ny = np.ones_like(depth) * 2.2
        nz = -dz
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx, ny, nz = nx/norm, ny/norm, nz/norm

        # Blinn-Phong lighting: Diffuse + Specular
        diffuse = np.maximum(0, nx*L[0] + ny*L[1] + nz*L[2])
        H = L + np.array([0, 1.0, 0])
        H = H / np.linalg.norm(H)
        specular = np.maximum(0, nx*H[0] + ny*H[1] + nz*H[2])**16

        intensity = (0.28 + 0.62 * diffuse + 0.22 * specular) * hit
        intensity = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

        # Apply anatomical LUT
        lut = get_color_lut(preset["palette"])
        color_view = cv2.LUT(cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR), lut)

        # Medical annotation overlay
        cv2.putText(color_view, f"3D VR {int(angle_deg)} deg | {preset_key}", (8, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        return color_view

    def render_mip_view(self, angle_deg=0.0, preset_key="VASCULAR_CTA_MIP"):
        """Renders 3D Maximum Intensity Projection (MIP) for CT Angiography."""
        if self.volume_iso is None:
            raise ValueError("No volume loaded.")

        preset = PRESETS.get(preset_key, PRESETS["VASCULAR_CTA_MIP"])
        threshold = preset["hu_threshold"]

        if angle_deg != 0.0:
            h, w = self.volume_iso.shape[1], self.volume_iso.shape[2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rot = np.empty_like(self.volume_iso)
            for z in range(self.volume_iso.shape[0]):
                rot[z] = cv2.warpAffine(self.volume_iso[z], M, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            rot = self.volume_iso

        vessels = np.where(rot > threshold, rot, 0.0)
        mip = np.max(vessels, axis=1)
        mip_norm = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        lut = get_color_lut(preset["palette"])
        color_view = cv2.LUT(cv2.cvtColor(mip_norm, cv2.COLOR_GRAY2BGR), lut)

        cv2.putText(color_view, f"3D MIP {int(angle_deg)} deg | {preset_key}", (8, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return color_view

    def generate_full_360_cine(self, output_dir, num_frames=24, preset_key="BONE_TRAUMA", progress_callback=None):
        """Renders a complete 360-degree rotating cine series and saves all frames."""
        os.makedirs(output_dir, exist_ok=True)
        angles = np.linspace(0, 360, num_frames, endpoint=False)
        frame_paths = []

        is_mip = (preset_key == "VASCULAR_CTA_MIP")

        for idx, ang in enumerate(angles):
            if is_mip:
                frame = self.render_mip_view(ang, preset_key)
            else:
                frame = self.render_shaded_view(ang, preset_key)

            frame_filename = f"3D_frame_{idx+1:03d}_{int(ang)}deg.jpg"
            frame_full_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_full_path, frame)
            frame_paths.append(frame_full_path)

            if progress_callback:
                pct = 50 + int(((idx + 1) / num_frames) * 45)
                progress_callback(pct, f"رندر فریم ۳D چرخشی {idx+1} از {num_frames} ({int(ang)} درجه)...")

        # Save standard orthogonal key clinical snapshots
        self.save_orthogonal_snapshots(output_dir, preset_key)

        if progress_callback:
            progress_callback(100, "تکمیل بازسازی ۳D و ذخیره سری چرخشی ۳۶۰ درجه.")

        return frame_paths

    def save_orthogonal_snapshots(self, output_dir, preset_key="BONE_TRAUMA"):
        """Saves AP (0°), Lateral (90°), Posterior (180°), and Oblique (45°) high-res snapshots."""
        os.makedirs(output_dir, exist_ok=True)
        key_angles = [("AP", 0), ("OBLIQUE", 45), ("LATERAL", 90), ("POSTERIOR", 180)]
        for label, ang in key_angles:
            view = self.render_shaded_view(ang, preset_key)
            cv2.imwrite(os.path.join(output_dir, f"KEY_3D_{label}.jpg"), view)

    def export_stl_mesh(self, output_stl_path, threshold=95):
        """Exports 3D surface mesh as an STL file for 3D printing of bone/fracture (optional)."""
        if self.volume_iso is None:
            raise ValueError("No volume loaded.")
        try:
            from skimage import measure
            import trimesh
            vol_smooth = np.empty_like(self.volume_iso)
            for z in range(self.volume_iso.shape[0]):
                vol_smooth[z] = cv2.GaussianBlur(self.volume_iso[z], (5, 5), 0.8)
            verts, faces, normals, values = measure.marching_cubes(vol_smooth, threshold, spacing=(1.0, 1.0, 1.0))
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            mesh.export(output_stl_path)
            return len(verts), len(faces)
        except ImportError:
            raise ImportError("برای خروجی فایل STL پرینتر سه‌بعدی، پکیج‌های scikit-image و trimesh مورد نیاز است.")
