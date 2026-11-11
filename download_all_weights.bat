@echo off
chcp 65001 >nul
title BoneSuppression AI - Offline Weights Downloader
color 0b
echo ===============================================================================
echo     دریافت خودکار وزن‌های مدل هوش مصنوعی جداسازی استخوان قفسه سینه
echo     Automated Model Weights Downloader (835 MB Total)
echo ===============================================================================
echo.

if not exist weights mkdir weights

echo [1/2] در حال دانلود وزن‌های مدل حذف استخوان (bone_suppression.ts - 417 MB)...
powershell -Command "Start-BitsTransfer -Source 'https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts' -Destination 'weights\bone_suppression.ts' -DisplayName 'Bone Suppression Model'"
if not exist weights\bone_suppression.ts (
    echo خطای دانلود با BITS. تلاش با curl...
    curl -L --retry 3 -C - "https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/bone_suppression.ts" -o "weights\bone_suppression.ts"
)

echo.
echo [2/2] در حال دانلود وزن‌های مدل ساختار ریه (lung_component_suppression.ts - 417 MB)...
powershell -Command "Start-BitsTransfer -Source 'https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/lung_component_suppression.ts' -Destination 'weights\lung_component_suppression.ts' -DisplayName 'Lung Component Model'"
if not exist weights\lung_component_suppression.ts (
    echo خطای دانلود با BITS. تلاش با curl...
    curl -L --retry 3 -C - "https://huggingface.co/qureaiorg/bone-suppression/resolve/main/weights/lung_component_suppression.ts" -o "weights\lung_component_suppression.ts"
)

echo.
echo ===============================================================================
echo بررسی وضعیت فایل‌ها:
if exist weights\bone_suppression.ts (
    echo [✓] weights\bone_suppression.ts موجود است.
) else (
    echo [✖] فایل bone_suppression.ts دانلود نشد!
)

if exist weights\lung_component_suppression.ts (
    echo [✓] weights\lung_component_suppression.ts موجود است.
) else (
    echo [✖] فایل lung_component_suppression.ts دانلود نشد!
)
echo ===============================================================================
echo دانلود به پایان رسید. اکنون می‌توانید برنامه را به صورت کاملاً آفلاین اجرا کنید.
pause
