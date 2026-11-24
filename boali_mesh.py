"""
boali_mesh.py (v2) - استخراج مش استخوان از حجم HU واقعی

mode="mask" (آزمایشی؛ روی قشر نازک ممکن است ترابکولار را وارد سطح کند): مقاوم برای استخوان با قشر نازک/کم‌تراکم (مچ پا، پا، سالمند)
    hysteresis (کم/زیاد) -> حذف اجسام بیرون از بدن (تخت/میله) -> closing کوچک -> پرکردن حفره‌های بسته
    -> سطح هموار از ماسک توپر (iso=0.5). تخلخل ترابکولار دیگر روی سطح نمی‌افتد.
mode="iso" (پیش‌فرض): ایزوسطح مستقیم روی HU + توپرکردن عمق داخلی.

ورودی: vol_hu (Z,Y,X) و spacing=(dz,dy,dx) mm   |   خروجی verts (z,y,x) mm و faces
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import csr_matrix, csgraph
from skimage import measure

try:
    import vtk
    from vtk.util import numpy_support
    _VTK = True
except Exception:
    _VTK = False


def hu_to_u8(hu, lo=-1000.0, hi=2000.0):
    return int(round((float(hu) - lo) / (hi - lo) * 255.0))


def u8_to_hu(v, lo=-1000.0, hi=2000.0):
    return float(v) / 255.0 * (hi - lo) + lo


def _ball(spacing, r_mm):
    rad = [max(int(np.ceil(r_mm / s)), 1) for s in spacing]
    zz, yy, xx = np.ogrid[-rad[0]:rad[0] + 1, -rad[1]:rad[1] + 1, -rad[2]:rad[2] + 1]
    return ((zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2) <= r_mm ** 2 + 1e-9


# ------------------------------------------------------------ پوشش بدن (حذف تخت/میله‌های بیرون از اندام)
def _body_mask(vol, spacing, air=-300.0, open_mm=4.0):
    dz, dy, dx = spacing
    fz = 1 if dz >= 1.0 else 2
    fx = 3
    small = vol[::fz, ::fx, ::fx] > air
    sp = (dz * fz, dy * fx, dx * fx)
    eroded = ndi.distance_transform_edt(small, sampling=sp) > open_mm      # opening مورفولوژیک با EDT
    if not eroded.any():
        return None
    body = ndi.distance_transform_edt(~eroded, sampling=sp) <= open_mm
    body &= small
    lab, n = ndi.label(body)
    if n == 0:
        return None
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    iz = np.minimum(np.arange(vol.shape[0]) // fz, body.shape[0] - 1)
    iy = np.minimum(np.arange(vol.shape[1]) // fx, body.shape[1] - 1)
    ix = np.minimum(np.arange(vol.shape[2]) // fx, body.shape[2] - 1)
    full = body[iz][:, iy][:, :, ix]
    return ndi.binary_dilation(full, iterations=3)


def _select_bone(vol, low, high, spacing, min_mm3=80.0, remove_bed=True, body_filter=True):
    thresh = float(low)
    mask = vol > thresh
    if not mask.any() or np.sum(mask) < 150:
        thresh = max(60.0, thresh - 40.0)
        mask = vol > thresh
    if not mask.any():
        return np.ones_like(vol, dtype=bool)

    lab, n = ndi.label(mask)
    if n == 0:
        return mask

    idx = np.arange(1, n + 1)
    counts = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    peak = np.concatenate([[-1e9], np.atleast_1d(ndi.maximum(vol, lab, idx))])

    h_thresh = min(float(high), 220.0)
    ok = (counts * float(np.prod(spacing)) >= min_mm3) & (peak >= h_thresh)
    ok[0] = False

    if not ok.any():
        ok = (counts * float(np.prod(spacing)) >= min_mm3)
        ok[0] = False

    if body_filter and ok.any():
        try:
            body = _body_mask(vol, spacing)
            if body is not None:
                inside = np.concatenate([[0.0], np.atleast_1d(ndi.sum(body, lab, idx))])
                cand_ok = ok & (inside >= 0.20 * np.maximum(counts, 1))
                if cand_ok.any():
                    ok = cand_ok
        except Exception:
            pass

    if remove_bed and ok.any():
        try:
            Z, Y, X = vol.shape
            cand_ok = ok.copy()
            for i, sl in enumerate(ndi.find_objects(lab), start=1):
                if sl is None or not cand_ok[i]:
                    continue
                # 1. حذف تخت اسکنر در انتهای کادر تخت
                if (sl[2].stop - sl[2].start) > 0.75 * X and (sl[1].stop - sl[1].start) < 0.08 * Y and sl[1].stop > 0.85 * Y:
                    cand_ok[i] = False
                # 2. حذف هوشمند آتل گچی خارجی (Plaster Splint / Cast Removal)
                # گچ خارجی در بیش از ۴۵٪ حاشیه محیطی‌اش در تماس با هوای اتاق (Air) است
                elif counts[i] > 10000:
                    comp = (lab == i)
                    dil = ndi.binary_dilation(comp, iterations=2)
                    ring = dil & ~comp
                    if ring.any():
                        air_frac = np.mean(vol[ring] < -350.0)
                        if air_frac > 0.45:
                            cand_ok[i] = False  # تشخیص قطعی آتل گچی خارجی و کسر آن
            if cand_ok.any():
                ok = cand_ok
        except Exception:
            pass

    if not ok.any():
        largest_idx = np.argmax(counts[1:]) + 1
        ok = np.zeros(n + 1, dtype=bool)
        ok[largest_idx] = True

    return ok[lab]


# ------------------------------------------------------------ هموارسازی/تمیزکاری مش
def _adjacency(faces, n):
    r = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    c = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    a = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
    a = a + a.T
    a.data[:] = 1.0
    return a


def taubin_smooth(verts, faces, iters=10, lam=0.5, mu=-0.53):
    a = _adjacency(faces, len(verts))
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv = (1.0 / deg).astype(np.float32)[:, None]
    v = verts.astype(np.float32).copy()
    for _ in range(iters):
        for k in (lam, mu):
            v += k * ((a @ v) * dinv - v)
    return v


def _drop_small_pieces(verts, faces, min_tri):
    n = len(verts)
    _, lab = csgraph.connected_components(_adjacency(faces, n), directed=False)
    tl = lab[faces[:, 0]]
    u, c = np.unique(tl, return_counts=True)
    faces = faces[np.isin(tl, u[c >= min_tri])]
    used = np.unique(faces)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def euler_characteristic(faces):
    v = len(np.unique(faces))
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return v - len(e) + len(faces)


# ------------------------------------------------------------ کاهش پالیگان (VTK)
def _to_vtk(verts, faces):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(verts.astype(np.float32), deep=True))
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(faces), numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    return pd


def _from_vtk(pd):
    v = numpy_support.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float32)
    f = numpy_support.vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return v, f.astype(np.int64)


def decimate(verts, faces, target_triangles, method="quadric"):
    if not _VTK or target_triangles is None or len(faces) <= target_triangles:
        return verts, faces
    red = 1.0 - float(target_triangles) / len(faces)
    pd = _to_vtk(verts, faces)
    if method == "quadric":
        d = vtk.vtkQuadricDecimation()
        d.SetInputData(pd)
        d.SetTargetReduction(red)
        d.VolumePreservationOn()
    else:
        d = vtk.vtkDecimatePro()
        d.SetInputData(pd)
        d.SetTargetReduction(min(red, 0.95))
        d.PreserveTopologyOn()
        d.SplittingOff()
        d.BoundaryVertexDeletionOff()
    d.Update()
    return _from_vtk(d.GetOutput())


# ------------------------------------------------------------ تابع اصلی
def extract_bone_mesh(vol_hu, spacing, hu_threshold=300.0, target_triangles=300000,
                      mode="iso", hu_low=None, hu_high=None, close_mm=0.6, smooth_mm=0.45,
                      smooth_sigma_mm=0.30, taubin_iters=10, min_component_mm3=800.0,
                      remove_bed=True, body_filter=True, fill_cavities=True, core_erode_mm=1.5,
                      decimate_method="quadric", log=None):
    say = log or (lambda *_: None)
    dz, dy, dx = [float(s) for s in spacing]
    vol_hu = np.asarray(vol_hu)
    if mode == "mask":
        low = float(hu_low if hu_low is not None else max(150.0, 0.6 * hu_threshold))
        high = float(hu_high if hu_high is not None else 1.15 * hu_threshold)
    else:
        low = high = float(hu_threshold)
    say("mode=%s  low=%.0f  high=%.0f HU" % (mode, low, high))

    keep = _select_bone(vol_hu, low, high, (dz, dy, dx), min_component_mm3, remove_bed, body_filter)
    if not keep.any():
        raise ValueError("بعد از فیلتر هیچ استخوانی نماند؛ آستانه را کم کنید یا body_filter=False بگذارید.")
    idx = np.where(keep)
    m = (int(np.ceil(6.0 / dz)), int(np.ceil(6.0 / dy)), int(np.ceil(6.0 / dx)))
    sl = tuple(slice(max(int(i.min()) - mm, 0), int(i.max()) + mm + 1) for i, mm in zip(idx, m))
    ks = keep[sl]
    f = int(np.clip(round(dz / (2.0 * min(dy, dx))), 1, 4))

    if mode == "mask":
        if close_mm > 0:
            ks = ndi.binary_closing(ks, structure=_ball((dz, dy, dx), close_mm))   # منافذ/نویز کوچک قشر را می‌بندد
        if fill_cavities:
            ks = ndi.binary_fill_holes(ks)                                        # حفره‌های کاملاً بسته‌ی داخلی
        vol = ks.astype(np.float32)
        level = 0.5
        sig = smooth_mm
    else:
        sub = vol_hu[sl].astype(np.float32)
        kd = ndi.binary_dilation(ks, iterations=2)
        if fill_cavities:
            sm = ndi.gaussian_filter(ks.astype(np.float32), (1.0 / dz, 1.0 / dy, 1.0 / dx)) > 0.5
            core = ndi.binary_fill_holes(sm)
            deep = ndi.distance_transform_edt(core, sampling=(dz, dy, dx)) > core_erode_mm
            sub[deep] = np.maximum(sub[deep], high + 200.0)
            kd |= deep
        sub[~kd] = -1000.0
        vol, level, sig = sub, high, smooth_sigma_mm

    if f > 1:
        vol = ndi.zoom(vol, (f, 1, 1), order=1, mode="nearest", grid_mode=True)
        dz = dz / f
    vol = ndi.gaussian_filter(vol, (sig / dz, sig / dy, sig / dx))
    say("حجم کراپ‌شده:", vol.shape, "spacing:", (round(dz, 3), dy, dx))

    verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=(dz, dy, dx), allow_degenerate=False)
    say("marching cubes:", len(faces), "مثلث")
    verts, faces = _drop_small_pieces(verts, faces, 200)
    verts = taubin_smooth(verts, faces, taubin_iters)
    n0 = len(faces)
    verts, faces = decimate(verts, faces, target_triangles, decimate_method)
    say("مثلث‌ها: %d -> %d%s" % (n0, len(faces), "" if _VTK else "  (VTK نصب نیست؛ ساده‌سازی انجام نشد)"))
    return verts.astype(np.float32), faces


# ------------------------------------------------------------ STL
def signed_volume(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def to_patient_xyz(verts_zyx, faces):
    """(z,y,x)->(x,y,z) تا مدل آینه‌ای نشود؛ winding طوری که normalها رو به بیرون باشند."""
    v = verts_zyx[:, ::-1].copy()
    f = faces.copy()
    if signed_volume(v, f) < 0:
        f = f[:, ::-1].copy()
    return v, f


def write_stl_xyz(path, v_xyz, faces, header=b"Boali 3D Bone Model"):
    """نقاط از قبل در مختصات بیمار (x,y,z) هستند؛ فقط winding را برای normal بیرونی اصلاح می‌کند."""
    v = np.asarray(v_xyz, dtype=np.float32)
    f = faces.copy()
    if signed_volume(v, f) < 0:
        f = f[:, ::-1].copy()
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3),
                                  ("c", "<f4", 3), ("attr", "<u2")])
    rec["n"], rec["a"], rec["b"], rec["c"] = nrm, v0, v1, v2
    with open(path, "wb") as fh:
        fh.write(header.ljust(80, b" ")[:80])
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())
    return path


def write_stl(path, verts_zyx, faces, header=b"Boali 3D Bone Model"):
    v, f = to_patient_xyz(verts_zyx, faces)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3),
                                  ("c", "<f4", 3), ("attr", "<u2")])
    rec["n"], rec["a"], rec["b"], rec["c"] = nrm, v0, v1, v2
    with open(path, "wb") as fh:
        fh.write(header.ljust(80, b" ")[:80])
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())
    return path
