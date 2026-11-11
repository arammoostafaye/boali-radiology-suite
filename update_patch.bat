@echo off
setlocal enabledelayedexpansion
title نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (v3.0.2)
color 0f

set "URL1=https://github.com/arammoostafaye/boali-radiology-suite/releases/download/v3.0.2/Boali_Radiology_Suite_v3.0.2_Update.zip"
set "URL2=https://github.com/arammoostafaye/boali-radiology-suite/releases/latest/download/Boali_Radiology_Suite_v3.0.2_Update.zip"
set "OUT=%~dp0Boali_Radiology_Suite_v3.0.2_Update.zip"

echo ======================================================================
echo   نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (v3.0.2)
echo   توسعه‌دهنده: آرام مصطفائی | سامانه تخصصی هوش مصنوعی و استودیوی سه‌بعدی
echo ======================================================================
echo.
echo  در حال دریافت پکیج سبک آپدیت نگارش v3.0.2...
echo  (فایل‌های حجیم وزن‌های هوش مصنوعی حفظ می‌شوند و نیازی به دانلود مجدد ندارند)
echo.

set N=0
:loop
set /a N+=1
if %N% GTR 12 (
  echo.
  echo  خطا در برقراری ارتباط پایدار. لطفاً اتصال اینترنت را چک فرمایید.
  pause
  exit /b 1
)
echo  در حال دریافت فایل با قابلیت ادامه (Resume) - تلاش %N% ...
curl -L -C - --retry 5 --retry-all-errors --retry-delay 2 --connect-timeout 20 -o "%OUT%" "%URL1%"
if errorlevel 1 curl -L -C - --retry 5 --retry-all-errors --retry-delay 2 --connect-timeout 20 -o "%OUT%" "%URL2%"

if not exist "%OUT%" goto loop

echo.
echo  [✓] دریافت کامل شد. در حال استخراج بروزرسانی...
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%OUT%' -DestinationPath '%~dp0' -Force"

if exist "%~dp0BoneSuppressionAI\BoneSuppressionAI.exe" (
    set "APPDIR=%~dp0BoneSuppressionAI"
    copy /y "%~dp0BoneSuppression_Desktop.py" "%~dp0BoneSuppressionAI\" >nul 2>&1
    copy /y "%~dp0boali_*.py" "%~dp0BoneSuppressionAI\" >nul 2>&1
    copy /y "%~dp0pacs_downloader.py" "%~dp0BoneSuppressionAI\" >nul 2>&1
    copy /y "%~dp0*.bat" "%~dp0BoneSuppressionAI\" >nul 2>&1
    copy /y "%~dp0config.json" "%~dp0BoneSuppressionAI\" >nul 2>&1
) else (
    set "APPDIR=%~dp0"
)

echo.
echo  [✓] اتصال و انتقال مدل‌های قبلی به پوشه برنامه...
cd /d "%APPDIR%"
if exist "apply_update.bat" call "apply_update.bat"
if exist "اجرای_نرم_افزار_بوعلی.bat" start "" "اجرای_نرم_افزار_بوعلی.bat"
exit /b 0
