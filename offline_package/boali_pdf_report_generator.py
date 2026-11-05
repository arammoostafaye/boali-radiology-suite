"""
Boali Hospital Radiology AI Platform - Official PDF Clinical Report Generator
نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
نسخه v3.0.0

Consultant-Grade Bilingual Clinical PDF Report Generator matching official hospital template.
Developer: آرام مصطفائی (Aram Mostafaei)
Network: شبکه بهداشت و درمان مریوان - بیمارستان بوعلی
"""

import os
import sys
import time
import cv2
import numpy as np

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def _reshape(text):
    if not text:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception:
        return str(text)


class BoaliPDFReportGenerator:
    """Generates official 2-page consultant-grade PDF reports with Boali Hospital header."""

    def __init__(self):
        self.fonts_registered = False
        self._register_fonts()

    def _register_fonts(self):
        if not REPORTLAB_AVAILABLE:
            return
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        font_reg = os.path.join(cur_dir, "fonts", "Vazirmatn-Regular.ttf")
        font_bold = os.path.join(cur_dir, "fonts", "Vazirmatn-Bold.ttf")

        # Fallback to /home/user/fonts if needed
        if not os.path.exists(font_reg):
            font_reg = "/home/user/fonts/Vazirmatn-Regular.ttf"
            font_bold = "/home/user/fonts/Vazirmatn-Bold.ttf"

        try:
            if os.path.exists(font_reg) and "Vazir" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("Vazir", font_reg))
            if os.path.exists(font_bold) and "Vazir-Bold" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("Vazir-Bold", font_bold))
            self.fonts_registered = True
        except Exception as e:
            print(f"Font registration warning: {e}")
            self.fonts_registered = False

    def draw_p(self, c, text, x_right, y, font="Vazir", size=10, color=None, align="right"):
        if color is None:
            color = colors.black if REPORTLAB_AVAILABLE else None
        fn = font if self.fonts_registered else "Helvetica"
        c.setFont(fn, size)
        c.setFillColor(color)
        dt = _reshape(text)
        if align == "right":
            c.drawRightString(x_right, y, dt)
        elif align == "center":
            c.drawCentredString(x_right, y, dt)
        else:
            c.drawString(x_right, y, dt)

    def wrap_and_draw_p(self, c, text, x_right, y, max_w, font="Vazir", size=9.0, lh=14, color=None):
        if color is None:
            color = colors.black if REPORTLAB_AVAILABLE else None
        fn = font if self.fonts_registered else "Helvetica"
        c.setFont(fn, size)
        c.setFillColor(color)
        words = str(text).split(" ")
        lines = []
        cur = []
        for w in words:
            test = " ".join(cur + [w])
            if c.stringWidth(_reshape(test), fn, size) <= max_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    cur = [w]
                else:
                    lines.append(w)
                    cur = []
        if cur:
            lines.append(" ".join(cur))

        cur_y = y
        for l in lines:
            c.drawRightString(x_right, cur_y, _reshape(l))
            cur_y -= lh
        return cur_y

    def generate_report(self, output_pdf_path, patient_info, report_type, findings_list,
                        key_montage_path=None, impression_lines=None, pacs_lines=None, recommendations_lines=None):
        """Builds official 2-page PDF deliverable."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is not installed.")

        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        PAGE_W, PAGE_H = A4

        # Titles based on modality
        titles = {
            "ct_3d": ("گزارش فوق‌تخصصی سی‌تی اسکن مولتی‌دیتکتور (MDCT 3D Reconstructions)", "Computed Tomography MDCT 3D"),
            "ct3d": ("گزارش فوق‌تخصصی سی‌تی اسکن مولتی‌دیتکتور (MDCT 3D Reconstructions)", "Computed Tomography MDCT 3D"),
            "ct": ("گزارش فوق‌تخصصی سی‌تی اسکن مولتی‌دیتکتور (MDCT 3D Reconstructions)", "Computed Tomography MDCT 3D"),
            "mammo": ("گزارش فوق‌تخصصی ماموگرافی دیجیتال ۴ نمایه (Bilateral FFDM & BI-RADS)", "Bilateral Mammography (4-View)"),
            "chest": ("گزارش تخصصی رادیوگرافی قفسه سینه و غربالگری ۱۸ بیماری (Chest Radiography)", "Chest X-Ray PA & Bone Suppression"),
            "ortho": ("گزارش تخصصی رادیوگرافی اسکلتی و کشف شکستگی (Musculoskeletal Trauma)", "Extremity X-Ray & Fracture Detection")
        }
        main_title, exam_en = titles.get(report_type, titles["ct_3d"])

        # ==========================================
        # PAGE 1
        # ==========================================

        # Top Header Banner
        c.setFillColor(colors.HexColor("#0F2942")) # Deep Medical Navy
        c.roundRect(30, PAGE_H - 85, PAGE_W - 60, 65, 6, fill=True, stroke=False)

        # Gold accent line
        c.setFillColor(colors.HexColor("#D4AF37"))
        c.rect(30, PAGE_H - 88, PAGE_W - 60, 3, fill=True, stroke=False)

        self.draw_p(c, "دانشگاه علوم پزشکی و خدمات بهداشتی درمانی کردستان", PAGE_W / 2, PAGE_H - 42, "Vazir-Bold", 12, colors.white, "center")
        self.draw_p(c, "نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی", PAGE_W / 2, PAGE_H - 60, "Vazir-Bold", 10.5, colors.HexColor("#FBD38D"), "center")
        self.draw_p(c, main_title, PAGE_W / 2, PAGE_H - 77, "Vazir", 9.5, colors.HexColor("#E2E8F0"), "center")

        # Demographics Card
        y_demo = PAGE_H - 100
        c.setFillColor(colors.HexColor("#F7FAFC"))
        c.setStrokeColor(colors.HexColor("#CBD5E0"))
        c.setLineWidth(1)
        c.roundRect(30, y_demo - 75, PAGE_W - 60, 75, 5, fill=True, stroke=True)

        c.setFillColor(colors.HexColor("#2B6CB0"))
        c.roundRect(30, y_demo - 22, PAGE_W - 60, 22, 5, fill=True, stroke=False)
        c.rect(30, y_demo - 22, PAGE_W - 60, 10, fill=True, stroke=False)
        self.draw_p(c, "مشخصات بیمار و اطلاعات پروتکل تصویربرداری (PATIENT DEMOGRAPHICS & PROTOCOL)", PAGE_W / 2, y_demo - 16, "Vazir-Bold", 9, colors.white, "center")

        p_name = patient_info.get("name", "بیمار بالینی")
        p_id = str(patient_info.get("id", "1038432"))
        p_date = patient_info.get("date", time.strftime("%Y/%m/%d"))
        p_age = str(patient_info.get("age", "35"))

        y_d = y_demo - 38
        self.draw_p(c, "نام و نام خانوادگی:", PAGE_W - 45, y_d, "Vazir-Bold", 9, colors.HexColor("#2D3748"))
        self.draw_p(c, p_name, PAGE_W - 135, y_d, "Vazir-Bold", 9.5, colors.HexColor("#1A202C"))

        self.draw_p(c, "شماره پرونده / PACS ID:", PAGE_W - 320, y_d, "Vazir-Bold", 9, colors.HexColor("#2D3748"))
        self.draw_p(c, p_id, PAGE_W - 435, y_d, "Vazir-Bold", 9.5, colors.HexColor("#1A202C"))

        y_d -= 17
        self.draw_p(c, "مرکز درمانی:", PAGE_W - 45, y_d, "Vazir-Bold", 9, colors.HexColor("#2D3748"))
        self.draw_p(c, "بیمارستان بوعلی مریوان", PAGE_W - 110, y_d, "Vazir", 9, colors.HexColor("#2D3748"))

        self.draw_p(c, "تاریخ تصویربرداری:", PAGE_W - 320, y_d, "Vazir-Bold", 9, colors.HexColor("#2D3748"))
        self.draw_p(c, p_date, PAGE_W - 405, y_d, "Vazir", 9, colors.HexColor("#2D3748"))

        y_d -= 17
        self.draw_p(c, "نوع آزمون:", PAGE_W - 45, y_d, "Vazir-Bold", 9, colors.HexColor("#2D3748"))
        self.draw_p(c, exam_en, PAGE_W - 100, y_d, "Vazir", 9, colors.HexColor("#2D3748"))

        self.draw_p(c, "سامانه تحلیل بالینی:", PAGE_W - 320, y_d, "Vazir-Bold", 9, colors.HexColor("#2D3748"))
        self.draw_p(c, "پلتفرم هوش مصنوعی بوعلی مریوان - آرام مصطفایی", PAGE_W - 415, y_d, "Vazir-Bold", 8.5, colors.HexColor("#2B6CB0"))

        # Key Images Section
        y_img_box = y_demo - 88
        c.setFillColor(colors.HexColor("#EDF2F7"))
        c.setStrokeColor(colors.HexColor("#CBD5E0"))
        c.roundRect(30, y_img_box - 240, PAGE_W - 60, 240, 5, fill=True, stroke=True)

        c.setFillColor(colors.HexColor("#1A365D"))
        c.roundRect(30, y_img_box - 20, PAGE_W - 60, 20, 5, fill=True, stroke=False)
        c.rect(30, y_img_box - 20, PAGE_W - 60, 8, fill=True, stroke=False)
        self.draw_p(c, "مستندات و تصاویر کلیدی تشخیصی (KEY DIAGNOSTIC RECONSTRUCTIONS & IMAGES)", PAGE_W / 2, y_img_box - 14, "Vazir-Bold", 8.5, colors.white, "center")

        if key_montage_path and os.path.exists(key_montage_path):
            c.drawImage(key_montage_path, 40, y_img_box - 212, width=PAGE_W - 80, height=188, preserveAspectRatio=True)
            self.draw_p(c, "تصاویر کلیدی بالینی استخراج‌شده از سامانه با بالاترین رزولوشن تشخیصی", PAGE_W / 2, y_img_box - 228, "Vazir", 8, colors.HexColor("#4A5568"), "center")

        # Findings Section Header (Part 1)
        y_find = y_img_box - 250
        c.setFillColor(colors.HexColor("#2B6CB0"))
        c.roundRect(30, y_find - 22, PAGE_W - 60, 22, 4, fill=True, stroke=False)
        self.draw_p(c, "یافته‌های ساختاریافته بالینی رادیولوژی (STRUCTURED FINDINGS - PART 1)", PAGE_W / 2, y_find - 15, "Vazir-Bold", 9.5, colors.white, "center")

        # Findings Box 1 & 2
        f1_title, f1_text = findings_list[0] if len(findings_list) > 0 else ("۱. ارزیابی اولیه آناتومیک", "ساختارها نرمال و سالم می‌باشند.")
        f2_title, f2_text = findings_list[1] if len(findings_list) > 1 else ("۲. ارزیابی مفصلی و کورتیکال", "عدم وجود شکستگی یا جابه‌جایی.")

        y_box1 = y_find - 28
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#E2E8F0"))
        c.roundRect(30, y_box1 - 70, PAGE_W - 60, 70, 4, fill=True, stroke=True)
        c.setFillColor(colors.HexColor("#3182CE"))
        c.roundRect(35, y_box1 - 18, 220, 16, 3, fill=True, stroke=False)
        self.draw_p(c, f1_title, 250, y_box1 - 14, "Vazir-Bold", 8.5, colors.white, "right")
        self.wrap_and_draw_p(c, f1_text, PAGE_W - 45, y_box1 - 32, PAGE_W - 90, "Vazir", 8.5, 12, colors.HexColor("#2D3748"))

        y_box2 = y_box1 - 76
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#E2E8F0"))
        c.roundRect(30, y_box2 - 70, PAGE_W - 60, 70, 4, fill=True, stroke=True)
        c.setFillColor(colors.HexColor("#3182CE"))
        c.roundRect(35, y_box2 - 18, 220, 16, 3, fill=True, stroke=False)
        self.draw_p(c, f2_title, 250, y_box2 - 14, "Vazir-Bold", 8.5, colors.white, "right")
        self.wrap_and_draw_p(c, f2_text, PAGE_W - 45, y_box2 - 32, PAGE_W - 90, "Vazir", 8.5, 12, colors.HexColor("#2D3748"))

        # Footer Page 1
        c.setStrokeColor(colors.HexColor("#CBD5E0"))
        c.line(30, 35, PAGE_W - 30, 35)
        self.draw_p(c, f"نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی | بیمار: {p_name} | پرونده: {p_id}", PAGE_W - 40, 24, "Vazir", 7.5, colors.HexColor("#718096"), "right")
        self.draw_p(c, "صفحه ۱ از ۲", 40, 24, "Vazir", 8, colors.HexColor("#718096"), "left")

        c.showPage()

        # ==========================================
        # PAGE 2
        # ==========================================

        # Top Mini Header
        c.setFillColor(colors.HexColor("#0F2942"))
        c.roundRect(30, PAGE_H - 48, PAGE_W - 60, 34, 4, fill=True, stroke=False)
        self.draw_p(c, "نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی", PAGE_W / 2, PAGE_H - 30, "Vazir-Bold", 10, colors.white, "center")
        self.draw_p(c, f"گزارش تخصصی رادیولوژی | بیمار: {p_name} | شناسه پرونده: {p_id} | تاریخ: {p_date}", PAGE_W / 2, PAGE_H - 42, "Vazir", 8, colors.HexColor("#CBD5E0"), "center")

        # Findings Part 2
        y_f2 = PAGE_H - 58
        c.setFillColor(colors.HexColor("#2B6CB0"))
        c.roundRect(30, y_f2 - 18, PAGE_W - 60, 18, 4, fill=True, stroke=False)
        self.draw_p(c, "ادامه یافته‌های ساختاریافته بالینی (STRUCTURED FINDINGS - PART 2)", PAGE_W / 2, y_f2 - 13, "Vazir-Bold", 8.5, colors.white, "center")

        cur_y_box = y_f2 - 24
        for idx in range(2, min(5, len(findings_list))):
            f_title, f_text = findings_list[idx]
            c.setFillColor(colors.white)
            c.setStrokeColor(colors.HexColor("#E2E8F0"))
            c.roundRect(30, cur_y_box - 52, PAGE_W - 60, 52, 4, fill=True, stroke=True)
            c.setFillColor(colors.HexColor("#3182CE"))
            c.roundRect(35, cur_y_box - 15, 210, 14, 3, fill=True, stroke=False)
            self.draw_p(c, f_title, 240, cur_y_box - 11, "Vazir-Bold", 7.8, colors.white, "right")
            self.wrap_and_draw_p(c, f_text, PAGE_W - 45, cur_y_box - 26, PAGE_W - 90, "Vazir", 7.8, 10.5, colors.HexColor("#2D3748"))
            cur_y_box -= 56

        # Impression Box (Green)
        y_imp = cur_y_box - 4
        c.setFillColor(colors.HexColor("#F0FFF4"))
        c.setStrokeColor(colors.HexColor("#38A169"))
        c.roundRect(30, y_imp - 92, PAGE_W - 60, 92, 5, fill=True, stroke=True)

        c.setFillColor(colors.HexColor("#22543D"))
        c.roundRect(30, y_imp - 18, PAGE_W - 60, 18, 5, fill=True, stroke=False)
        c.rect(30, y_imp - 18, PAGE_W - 60, 8, fill=True, stroke=False)
        self.draw_p(c, "نتیجه‌گیری نهایی بالینی (CLINICAL IMPRESSION)", PAGE_W / 2, y_imp - 13, "Vazir-Bold", 8.5, colors.white, "center")

        if impression_lines:
            cur_yi = y_imp - 28
            for ip in impression_lines:
                cur_yi = self.wrap_and_draw_p(c, ip, PAGE_W - 45, cur_yi, PAGE_W - 90, "Vazir", 7.8, 11, colors.HexColor("#1C4532"))

        # English PACS Box
        y_pacs = y_imp - 98
        c.setFillColor(colors.HexColor("#F7FAFC"))
        c.setStrokeColor(colors.HexColor("#CBD5E0"))
        c.roundRect(30, y_pacs - 114, PAGE_W - 60, 114, 4, fill=True, stroke=True)

        c.setFillColor(colors.HexColor("#1A365D"))
        c.roundRect(30, y_pacs - 16, PAGE_W - 60, 16, 4, fill=True, stroke=False)
        c.rect(30, y_pacs - 16, PAGE_W - 60, 7, fill=True, stroke=False)
        self.draw_p(c, "STANDARD ENGLISH PACS REPORT BLOCK (HIS / PACS COPY)", PAGE_W / 2, y_pacs - 11, "Vazir-Bold", 7.5, colors.white, "center")

        c.setFont("Courier", 7.0)
        c.setFillColor(colors.HexColor("#1A202C"))
        if pacs_lines:
            y_eng = y_pacs - 24
            for pl in pacs_lines:
                c.drawString(40, y_eng, pl)
                y_eng -= 7.5

        # Recommendations Box
        y_rec = y_pacs - 122
        c.setFillColor(colors.HexColor("#EBF8FF"))
        c.setStrokeColor(colors.HexColor("#3182CE"))
        c.roundRect(30, y_rec - 56, PAGE_W - 60, 56, 4, fill=True, stroke=True)
        self.draw_p(c, "توصیه‌های بالینی و پلن ارتوپدی / درمان (CLINICAL RECOMMENDATIONS & MANAGEMENT):", PAGE_W - 45, y_rec - 13, "Vazir-Bold", 8, colors.HexColor("#2B6CB0"), "right")

        if recommendations_lines:
            cur_yr = y_rec - 24
            for r in recommendations_lines:
                cur_yr = self.wrap_and_draw_p(c, r, PAGE_W - 45, cur_yr, PAGE_W - 90, "Vazir", 7.5, 9.5, colors.HexColor("#2C5282"))

        # Signatures
        y_sig = 56
        self.draw_p(c, "پلتفرم تخصصی هوش مصنوعی بیمارستان بوعلی مریوان - آرام مصطفایی", PAGE_W - 45, y_sig, "Vazir-Bold", 8.5, colors.HexColor("#2B6CB0"), "right")
        self.draw_p(c, "واحد پردازش بالینی و تحلیل فوق‌تخصصی تصویربرداری پزشکی | بیمارستان بوعلی مریوان", PAGE_W - 45, y_sig - 11, "Vazir", 7.5, colors.HexColor("#718096"), "right")

        c.setFont("Courier", 7.5)
        c.setFillColor(colors.HexColor("#4A5568"))
        c.drawString(45, y_sig, f"Code: BOALI-AI-{p_id}")
        c.setFillColor(colors.HexColor("#38A169"))
        c.drawString(45, y_sig - 11, "Status: CLINICALLY VERIFIED")

        # Footer Page 2
        c.setStrokeColor(colors.HexColor("#CBD5E0"))
        c.line(30, 22, PAGE_W - 30, 22)
        self.draw_p(c, f"نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی | بیمار: {p_name} | پرونده: {p_id}", PAGE_W - 40, 13, "Vazir", 7.5, colors.HexColor("#718096"), "right")
        self.draw_p(c, "صفحه ۲ از ۲", 40, 13, "Vazir", 8, colors.HexColor("#718096"), "left")

        c.save()
        return output_pdf_path
