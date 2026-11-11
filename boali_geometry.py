"""
boali_geometry.py - بازسازی حجم متعامد در مختصات استاندارد بالینی بیمار (DICOM LPS)
توسعه: آرام مصطفائی | واحد تصویربرداری بیمارستان بوعلی مریوان

این ماژول مقاطع سی‌تی اسکن را (چه محوری Axial، چه تاجی Coronal، چه ساجیتال یا مایل با شیب گانتری)
به حجم متعامد استاندارد بالینی (Z=Superior, Y=Posterior, X=Left) تبدیل می‌کند.
در نتیجه:
  - سر بیمار همیشه در بالا (+Z) و پاها در پایین قرار می‌گیرند.
  - جلوی بدن در سمت قدامی (-Y) و ستون فقرات/تخت در سمت خلفی (+Y) قرار می‌گیرند.
  - نمای قدامی (AP View 0°) همیشه مستقیم از روبه‌رو و بدون اعوجاج رندر می‌شود.
"""
import numpy as np
import scipy.ndimage as ndi


def make_geometry(ipp, iop, pixel_spacing):
    """ipp: (N,3) موقعیت مقاطع | iop: (6,) بردارهای ردیف و ستون | pixel_spacing: (dy, dx)"""
    ipp = np.asarray(ipp, dtype=np.float64)
    iop = np.asarray(iop, dtype=np.float64)
    Fr = iop[:3] / (np.linalg.norm(iop[:3]) + 1e-12)
    Fc = iop[3:] / (np.linalg.norm(iop[3:]) + 1e-12)
    n = np.cross(Fr, Fc)
    N = len(ipp)
    k = np.arange(N, dtype=np.float64)
    S = np.array([np.polyfit(k, ipp[:, i], 1)[0] for i in range(3)])
    dy, dx = float(pixel_spacing[0]), float(pixel_spacing[1])
    A = np.stack([S, dy * Fc, dx * Fr], axis=1)

    pos = ipp @ n
    d = np.diff(pos)
    dz_perp = float(abs(S @ n))
    if dz_perp < 0.1:
        dz_perp = float(np.linalg.norm(S)) or 1.25

    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(S @ n) / (np.linalg.norm(S) + 1e-12), -1, 1))))
    irregular = bool(len(d) and (np.abs(d).max() > 1.2 * (np.median(np.abs(d)) + 1e-6)))

    return dict(Fr=Fr, Fc=Fc, n=n, S=S, A=A, O=ipp[0], dy=dy, dx=dx, N=N,
                dz_perp=dz_perp, tilt_deg=tilt_deg, irregular=irregular)


def make_grid(g, H, W, dz_out=None):
    """ایجاد شبکه‌ی متعامد در مختصات استاندارد بالینی LPS (Z, Y, X)."""
    N = g["N"]
    corners_krc = np.array([[k, r, c] for k in (0, N - 1) for r in (0, H - 1) for c in (0, W - 1)], dtype=np.float64)
    pts_lps = corners_krc @ g["A"].T

    x_min, x_max = float(pts_lps[:, 0].min()), float(pts_lps[:, 0].max())
    y_min, y_max = float(pts_lps[:, 1].min()), float(pts_lps[:, 1].max())
    z_min, z_max = float(pts_lps[:, 2].min()), float(pts_lps[:, 2].max())

    target_dz = float(dz_out if dz_out is not None else min(g["dz_perp"], 1.5))
    target_dz = max(0.5, target_dz)
    sp = (target_dz, g["dy"], g["dx"])

    shape = (
        max(int(np.ceil((z_max - z_min) / sp[0])) + 1, 4),
        max(int(np.ceil((y_max - y_min) / sp[1])) + 1, 4),
        max(int(np.ceil((x_max - x_min) / sp[2])) + 1, 4),
    )
    origin = (z_min, y_min, x_min)
    return dict(origin=origin, spacing=sp, shape=shape, bounds=((z_min, z_max), (y_min, y_max), (x_min, x_max)))


def sub_grid(grid, lo_idx, hi_idx, dz_out=None):
    """زیرشبکه (ROI) با شاخص‌های lo..hi در مختصات (Z, Y, X)."""
    sp = list(grid["spacing"])
    org = np.asarray(grid["origin"], dtype=np.float64) + np.asarray(lo_idx) * np.asarray(sp)
    ext = (np.asarray(hi_idx) - np.asarray(lo_idx)) * np.asarray(sp)
    if dz_out is not None:
        sp[0] = float(dz_out)
    shape = tuple(max(int(np.ceil(ext[i] / sp[i])) + 1, 4) for i in range(3))
    return dict(origin=tuple(org), spacing=tuple(sp), shape=shape)


def resample(slices, g, grid, order=1, cval=-1000.0):
    """نمونه‌برداری متعامد slices به حجم استاندارد (Z, Y, X) در مختصات بالینی بیمار."""
    sp = grid["spacing"]
    z_min, y_min, x_min = grid["origin"]
    Ainv = np.linalg.inv(g["A"])

    M_out = np.array([
        [0.0, 0.0, sp[2]],
        [0.0, sp[1], 0.0],
        [sp[0], 0.0, 0.0],
    ], dtype=np.float64)

    affine_mat = Ainv @ M_out
    affine_offset = Ainv @ np.array([x_min, y_min, z_min], dtype=np.float64)

    return ndi.affine_transform(
        np.asarray(slices, dtype=np.float32),
        affine_mat,
        offset=affine_offset,
        output_shape=grid["shape"],
        order=order,
        mode="constant",
        cval=cval
    )


def uvw_to_patient(verts_zyx, g):
    """نقاط (Z, Y, X) به مختصات LPS بیمار."""
    return np.ascontiguousarray(verts_zyx[:, [2, 1, 0]], dtype=np.float32)
