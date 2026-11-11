@echo off
setlocal enabledelayedexpansion
title نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی (v3.0.2)
color 0f

echo ======================================================================
echo   نرم‌افزار تخصصی واحد تصویربرداری بیمارستان بوعلی - آرام مصطفائی
echo   بسته‌ساز و به‌روزرسان جامع نگارش v3.0.2
echo   شبکه بهداشت و درمان مریوان - بیمارستان بوعلی
echo ======================================================================
echo.

set "TARGET_DIR=%~dp0_internal\weights"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

if exist "%TARGET_DIR%\bone_suppression.ts" (
    echo [OK] وزن‌های مدل حذف استخوان در این پوشه موجود هستند.
    goto check_new_models
)

echo در حال جستجوی وزن‌های مدل در پوشه‌های نسخه قبلی...

set "FOUND="
for %%P in (
    "%~dp0..\_internal\weights"
    "%~dp0..\weights"
    "%~dp0..\BoneSuppressionAI\_internal\weights"
    "%~dp0..\BoneSuppressionAI\weights"
    "%~dp0..\..\BoneSuppressionAI\_internal\weights"
    "%~dp0..\..\weights"
    "%USERPROFILE%\Downloads\BoneSuppressionAI\_internal\weights"
    "%USERPROFILE%\Downloads\BoneSuppressionAI\weights"
    "%USERPROFILE%\Desktop\BoneSuppressionAI\_internal\weights"
    "C:\BoneSuppressionAI\_internal\weights"
    "D:\BoneSuppressionAI\_internal\weights"
    "E:\BoneSuppressionAI\_internal\weights"
    "E:\BoneSuppression-Windows\BoneSuppressionAI-Windows-x64-OfflineV2.2\BoneSuppressionAI\_internal\weights"
    "E:\BoneSuppression-Windows\BoneSuppressionAI-Windows-x64-OfflineV2.1.1\BoneSuppressionAI\_internal\weights"
) do (
    if exist "%%~fP\bone_suppression.ts" (
        set "FOUND=%%~fP"
        goto copy_weights
    )
)

:prompt_manual
echo.
echo [!] مسیر خودکار فایل‌های قبلی یافت نشد.
echo اگر پوشه weights نسخه قبلی را دارید، آن را درون پوشه _internal کپی نمایید.
goto check_new_models

:copy_weights
echo.
echo [✓] وزن‌های مدل قبلی در مسیر زیر یافت شدند:
echo     !FOUND!
echo.
echo در حال کپی مدل‌های سنگین ۸۳۵ مگابایتی به نسخه جدید...
copy /y "!FOUND!\bone_suppression.ts" "%TARGET_DIR%\" >nul
copy /y "!FOUND!\lung_component_suppression.ts" "%TARGET_DIR%\" >nul
echo [✓] انتقال وزن‌های قبلی با موفقیت انجام شد!
echo.

:check_new_models
echo [✓] بررسی مدل‌های ماموگرافی، شکستگی و بیماری‌های ریه...
if not exist "%TARGET_DIR%\mammo_resnet50.pt" (
    if exist "%~dp0weights\mammo_resnet50.pt" copy /y "%~dp0weights\mammo_resnet50.pt" "%TARGET_DIR%\" >nul
)
if not exist "%TARGET_DIR%\fracture_ortho.onnx" (
    if exist "%~dp0weights\fracture_ortho.onnx" copy /y "%~dp0weights\fracture_ortho.onnx" "%TARGET_DIR%\" >nul
)
if not exist "%TARGET_DIR%\fracture_yolov8.onnx" (
    if exist "%~dp0weights\fracture_yolov8.onnx" copy /y "%~dp0weights\fracture_yolov8.onnx" "%TARGET_DIR%\" >nul
)
if not exist "%TARGET_DIR%\cxr_densenet18.ts" (
    if exist "%~dp0weights\cxr_densenet18.ts" copy /y "%~dp0weights\cxr_densenet18.ts" "%TARGET_DIR%\" >nul
)

:copy_scripts_to_internal
if exist "%~dp0_internal" (
    echo [✓] همگام‌سازی کدهای پایتون و فونت‌ها در پوشه _internal...
    if exist "%~dp0BoneSuppression_Desktop.py" copy /y "%~dp0BoneSuppression_Desktop.py" "%~dp0_internal\" >nul
    if exist "%~dp0boali_ct_3d_engine.py" copy /y "%~dp0boali_ct_3d_engine.py" "%~dp0_internal\" >nul
    if exist "%~dp0boali_ct_3d_studio_dialog.py" copy /y "%~dp0boali_ct_3d_studio_dialog.py" "%~dp0_internal\" >nul
    if exist "%~dp0boali_pdf_report_generator.py" copy /y "%~dp0boali_pdf_report_generator.py" "%~dp0_internal\" >nul
    if exist "%~dp0pacs_downloader.py" copy /y "%~dp0pacs_downloader.py" "%~dp0_internal\" >nul
    if exist "%~dp0boali_mesh.py" copy /y "%~dp0boali_mesh.py" "%~dp0_internal\" >nul
    if exist "%~dp0boali_geometry.py" copy /y "%~dp0boali_geometry.py" "%~dp0_internal\" >nul
    if exist "%~dp0boali_ai_3d_segmentor.py" copy /y "%~dp0boali_ai_3d_segmentor.py" "%~dp0_internal\" >nul
    if exist "%~dp0fonts" xcopy /y /e /i "%~dp0fonts" "%~dp0_internal\fonts" >nul
)

:launch
echo ======================================================================
echo بررسی پیش‌نیازهای رندر سه‌بعدی و اجرای نرم‌افزار تخصصی واحد تصویربرداری بوعلی...
echo ======================================================================

python --version >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import vtk" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [اطلاع] نصب ماژول رندر سه‌بعدی پیشرفته بیمارستانی (VTK)...
        pip install vtk >nul 2>&1
    )
    echo [اطلاع] اجرای مستقیم آخرین نگارش پایتون بدون نیاز به کامپایل مجدد...
    start "" pythonw "%~dp0BoneSuppression_Desktop.py"
    exit /b 0
)

if exist "%~dp0BoneSuppressionAI.exe" (
    start "" "%~dp0BoneSuppressionAI.exe"
    exit /b 0
)

if exist "%~dp0dist\BoneSuppressionAI\BoneSuppressionAI.exe" (
    start "" "%~dp0dist\BoneSuppressionAI\BoneSuppressionAI.exe"
    exit /b 0
)

echo [اطلاع] برنامه در حال اجراست...
exit /b 0
