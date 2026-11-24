@echo off
chcp 65001 >nul
echo ========================================================
echo   همگام‌سازی و پوش تغییرات به مخزن رسمی گیت‌هاب بیمارستان بوعلی
echo   GitHub Repository: arammoostafaye/boali-radiology-suite
echo ========================================================
echo.

git status
echo.
echo در حال ارسال کامیت‌ها و تگ نگارش به گیت‌هاب...
git push origin main
git push origin --tags

echo.
echo ========================================================
echo   عملیات پوش با موفقیت انجام شد!
echo ========================================================
pause
