@echo off
chcp 65001 > nul
title Q-INTEGRITY - CARGANDO...
cd /d "%~dp0"
start /min python run_app.py
