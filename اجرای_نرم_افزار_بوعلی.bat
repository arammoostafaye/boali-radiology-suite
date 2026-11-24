@echo off
chcp 65001 >nul
title نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (v3.0.2)
color 0a

echo ===============================================================================
echo   نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
echo   نگارش پایدار v3.0.2
echo   شبکه بهداشت و درمان مریوان - بیمارستان بوعلی
echo ===============================================================================
echo.

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [اطلاع] اجرای مستقیم آخرین نگارش به‌روزشده با مفسر پایتون بدون پنجره کنسول...
    start "" pythonw "%~dp0BoneSuppression_Desktop.py"
    exit /b 0
)

if exist "%~dp0BoneSuppressionAI.exe" (
    echo [اطلاع] اجرای فایل اجرایی BoneSuppressionAI.exe...
    start "" "%~dp0BoneSuppressionAI.exe"
    exit /b 0
)

if exist "%~dp0dist\BoneSuppressionAI\BoneSuppressionAI.exe" (
    echo [اطلاع] اجرای فایل اجرایی dist\BoneSuppressionAI...
    start "" "%~dp0dist\BoneSuppressionAI\BoneSuppressionAI.exe"
    exit /b 0
)

echo [خطا] مفسر پایتون یا فایل اجرایی یافت نشد!
pause
exit /b 1
