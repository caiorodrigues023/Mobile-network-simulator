@echo off
title Mobile Network Simulator 3D - Modo Ultraleve
cd /d "%~dp0"

echo ========================================================
echo   Mobile Network Simulator 3D - Modo Ultraleve Web
echo ========================================================
echo Iniciando servidor local ultraleve (~35 MB RAM)...
echo O navegador padrao (Edge/Chrome) sera aberto automaticamente.
echo.

set PYTHONPATH=%~dp0;%PYTHONPATH%

rem 1. Procura primeiro pelos caminhos reais do Python instalado
if exist "C:\Users\Caio\AppData\Local\Python\pythoncore-3.15-64\python.exe" (
    "C:\Users\Caio\AppData\Local\Python\pythoncore-3.15-64\python.exe" app_web.py
    goto check_exit
)

if exist "C:\Users\Caio\AppData\Local\Python\bin\python.exe" (
    "C:\Users\Caio\AppData\Local\Python\bin\python.exe" app_web.py
    goto check_exit
)

rem 2. Tenta o inicializador 'py'
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3 app_web.py
    goto check_exit
)

rem 3. Tenta o python do PATH
python app_web.py
goto check_exit

:check_exit
if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================================
    echo [AVISO] O servidor encerrou com o codigo %ERRORLEVEL%.
    echo ========================================================
    echo.
    pause
)
