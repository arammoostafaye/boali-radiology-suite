@echo off
setlocal enabledelayedexpansion
title BoneSuppression AI - Full Offline Installer (v2.3.0)
color 0f

set "URL1=https://github.com/arammoostafaye/bone-suppression-chest-xray/releases/download/v2.3.0/BoneSuppressionAI-Windows-x64-Offline.zip"
set "URL2=https://huggingface.co/arammoostafaye/bone-suppression-desktop/resolve/main/BoneSuppressionAI-Windows-x64-Offline.zip"
set "OUT=%~dp0BoneSuppressionAI-Windows-x64-Offline.zip"

echo ======================================================================
echo   BoneSuppression AI - نصب و راه‌اندازی نسخه کامل و آفلاین v2.3.0
echo   شامل ۳ موتور هوش مصنوعی: تفکیک بافت نرم، شکستگی، ۱۸ بیماری ریه
echo   بیمارستان بوعلی مریوان - شبکه بهداشت و درمان مریوان
echo ======================================================================
echo.

if exist "%~dp0BoneSuppressionAI\BoneSuppressionAI.exe" (
    echo [OK] نرم‌افزار قبلاً نصب شده است. در حال اجرا...
    start "" "%~dp0BoneSuppressionAI\BoneSuppressionAI.exe"
    exit /b 0
)

echo در حال دریافت پکیج کامل آفلاین (~1 گیگابایت)...
set N=0
:loop
set /a N+=1
if %N% GTR 10 (
  echo خطا در دانلود. لطفاً اتصال اینترنت را بررسی فرمایید.
  pause
  exit /b 1
)
curl -L -C - --retry 5 --retry-all-errors --connect-timeout 20 -o "%OUT%" "%URL1%"
if errorlevel 1 curl -L -C - --retry 5 --retry-all-errors --connect-timeout 20 -o "%OUT%" "%URL2%"

if not exist "%OUT%" goto loop

echo.
echo [✓] دانلود کامل شد. در حال استخراج فایل‌ها...
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%OUT%' -DestinationPath '%~dp0' -Force"

if exist "%~dp0BoneSuppressionAI\BoneSuppressionAI.exe" (
    start "" "%~dp0BoneSuppressionAI\BoneSuppressionAI.exe"
)
exit /b 0
