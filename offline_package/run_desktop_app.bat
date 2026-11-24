@echo off
chcp 65001 >nul
title Launch Boali Radiology Suite AI Desktop v3.0.2
color 0f

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [اطلاع] اجرای مستقیم آخرین نسخه پایتون نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی (v3.0.2)...
    start "" pythonw BoneSuppression_Desktop.py
    exit /b 0
)

if exist "BoneSuppressionAI.exe" (
    start "" "BoneSuppressionAI.exe"
    exit /b 0
)

if exist "BoneSuppressionAI\BoneSuppressionAI.exe" (
    start "" "BoneSuppressionAI\BoneSuppressionAI.exe"
    exit /b 0
)

if exist "dist\BoneSuppressionAI\BoneSuppressionAI.exe" (
    start "" "dist\BoneSuppressionAI\BoneSuppressionAI.exe"
    exit /b 0
)

echo [خطا] پایتون یا فایل اجرایی یافت نشد.
pause
exit /b 1
