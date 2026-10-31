@echo off
title instatagger
rem Double-click to run the app at http://localhost:8765; it stays up until you close this window
cd /d "%~dp0"
python serve.py
pause
