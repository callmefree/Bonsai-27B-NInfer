@echo off
chcp 65001 >nul
title NInfer Launcher
rem ============================================================
rem  NInfer launcher entry. Double-click to open the menu.
rem  Uses system python at D:\Python311 (the 'python' command is
rem  NOT on PATH, so we call it by absolute path).
rem  CLI usage:
rem    python ninfer_launcher.py                 interactive menu
rem    python ninfer_launcher.py <profile>       launch named combo
rem    python ninfer_launcher.py --list          list saved combos
rem    python ninfer_launcher.py --dry-run <p>   print command only
rem ============================================================
cd /d J:\Bonsai
"D:\Python311\python.exe" ninfer_launcher.py
echo.
echo [launcher exited] press any key to close.
pause >nul