@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===============================================================================
echo   LIVE INTERACTIVE DEMO — AUTONOMOUS FRONTIER EXPLORATION ^& RESCUE ROBOT
echo   Ho tro Map A / B / C / D + Sinh Random Map ngau nhien (30 FPS sieu muot)
echo ===============================================================================
python demo_live_interactive.py
pause
