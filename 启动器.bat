@echo off
chcp 65001 >nul
rem ============================================================
rem  NInfer launcher. Double-click to open the GUI.
rem  GUI (no args): start pythonw via `start`, so the bat's own
rem  CMD window closes immediately (no blank window lingers).
rem  CLI (profile/--list/--dry-run): run python so output shows.
rem ============================================================
cd /d J:\Bonsai
if "%~1"=="" (
    start "NInfer-Launcher" "D:\Python311\pythonw.exe" "J:\Bonsai\ninfer_launcher.py"
) else (
    "D:\Python311\python.exe" ninfer_launcher.py %*
    echo.
    echo [launcher exited] press any key to close.
    pause >nul
)