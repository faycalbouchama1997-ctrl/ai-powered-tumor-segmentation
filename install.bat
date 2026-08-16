@echo off
chcp 65001 > nul

cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.10+ is required. Download it from python.org
    pause
    exit /b 1
)

python --version

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

echo.
echo Place model weights in models\weights\ -- see README.md for download links.
echo.
pause
