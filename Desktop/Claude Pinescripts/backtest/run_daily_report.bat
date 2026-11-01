@echo off
cd /d "%~dp0"
python daily_report.py >> logs\scheduled.log 2>&1
