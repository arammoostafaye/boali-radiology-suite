@echo off
chcp 65001 >nul
title Build BoneSuppression Standalone Windows EXE
color 0a
echo ===============================================================================
echo     کامپایل برنامه به فایل اجرایی مستقل ویندوز (.exe) با PyInstaller
echo     Compiling Standalone Windows Executable
echo ===============================================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [خطا] پایتون روی سیستم شما نصب نیست یا در PATH قرار ندارد!
    echo لطفاً پایتون 3.10 یا 3.11 را از python.org نصب و تیک Add to PATH را بزنید.
    pause
    exit /b 1
)

echo [1/3] بررسی و نصب کتابخانه‌های مورد نیاز...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo [2/3] بررسی وجود وزن‌های مدل...
if not exist weights\bone_suppression.ts (
    echo [هشدار] وزن‌های مدل در پوشه weights یافت نشدند!
    echo در حال اجرای خودکار دانلودر وزن‌ها...
    call download_all_weights.bat
)

echo.
echo [3/3] اجرای کامپایلر PyInstaller...
pyinstaller --noconfirm --onedir --windowed ^
    --name "BoneSuppressionAI" ^
    --add-data "weights;weights" ^
    --add-data "config.json;." ^
    --collect-all torch ^
    --collect-all cv2 ^
    BoneSuppression_Desktop.py

echo.
echo ===============================================================================
if exist dist\BoneSuppressionAI\BoneSuppressionAI.exe (
    echo [موفقیت] فایل اجرایی مستقل با موفقیت ساخته شد!
    echo مسیر فایل اجرایی: dist\BoneSuppressionAI\BoneSuppressionAI.exe
    echo می‌توانید پوشه dist\BoneSuppressionAI را روی هر سیستم ویندوز بدون اینترنت کپی و اجرا کنید.
) else (
    echo [خطا] در فرآیند ساخت فایل exe مشکلی پیش آمد. گزارش خطا را در بالا بررسی کنید.
)
echo ===============================================================================
pause
