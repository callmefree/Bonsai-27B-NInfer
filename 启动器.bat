@echo off
chcp 65001 >nul
title NInfer Launcher
rem ============================================================
rem  NInfer launcher entry. Double-click to open the GUI.
rem  No args -> run the GUI with pythonw (no blank CMD window).
rem  CLI args (profile/--list/--dry-run) -> use python so output
rem  stays visible in this console window.
rem ============================================================
cd /d J:\Bonsai
if "%~1"=="" (
    "D:\Python311\pythonw.exe" ninfer_launcher.py
) else (
    "D:\Python311\python.exe" ninfer_launcher.py %*
    echo.
    echo [launcher exited] press any key to close.
    pause >nul
)