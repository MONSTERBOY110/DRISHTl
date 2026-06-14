@echo off
REM DRISTI - Lost Person Detection System
REM Startup Script for Windows

echo.
echo ======================================================================
echo DRISTI - Lost Person Detection System v2.0
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check if CCTVS folder exists
if not exist "CCTVS" (
    echo.
    echo INFO: CCTVS folder not found
    echo To use video files, add MP4 videos to the CCTVS folder
    echo.
)

REM Check if data folder exists
if not exist "data" (
    mkdir data
)
if not exist "data\uploads" (
    mkdir data\uploads
)
if not exist "data\results" (
    mkdir data\results
)

REM Start the server
echo.
echo Starting DRISTI Backend Server...
echo Server will be available at: http://localhost:8000
echo.
python run.py
