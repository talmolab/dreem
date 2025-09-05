@echo off
setlocal enabledelayedexpansion

REM DREEM Installation Script for Windows
REM This script helps users install DREEM with the appropriate dependencies for their platform

echo 🚀 DREEM Installation Script
echo ==============================

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ uv is not installed. Please install uv first:
    echo    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

echo ✅ uv is installed

REM Detect architecture
for /f "tokens=*" %%i in ('wmic os get osarchitecture /value ^| find "="') do set %%i
echo 🔍 Detected architecture: %OSArchitecture%

REM Ask user for CUDA preference
echo 🐧 Windows detected. Do you want CUDA support? (y/n)
set /p cuda_choice=
if /i "%cuda_choice%"=="y" (
    set INSTALL_TYPE=cuda
    echo 🚀 Installing with CUDA support
) else (
    set INSTALL_TYPE=cpu
    echo 💻 Installing CPU-only version
)

REM Install dependencies
echo 📦 Installing dependencies...
uv sync --extra %INSTALL_TYPE%

REM Ask if user wants dev dependencies
echo 🔧 Do you want to install development dependencies? (y/n)
set /p dev_choice=
if /i "%dev_choice%"=="y" (
    echo 📦 Installing development dependencies...
    uv sync --extra dev --extra %INSTALL_TYPE%
)

echo ✅ Installation complete!
echo.
echo 🎯 To activate the environment, run:
echo    .venv\Scripts\activate
echo.
echo 🎯 Or use uv run to execute commands directly:
echo    uv run dreem-train --help
echo.
echo 🎯 To run DREEM commands:
echo    dreem-train --help
echo    dreem-track --help
echo    dreem-eval --help
echo    dreem-visualize --help
pause
