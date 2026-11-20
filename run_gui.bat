@echo off
title Mobile Network Simulator 3D
cd /d "%~dp0"

echo ========================================================
echo   Mobile Network Simulator 3D - Interface Grafica
echo ========================================================
echo Inicializando o ambiente e verificando instalacao do Python...

set PYTHONPATH=%~dp0;%PYTHONPATH%

rem Otimizacoes para GPUs integradas e computadores mais modestos
set QTWEBENGINE_CHROMIUM_FLAGS=--enable-webgl --ignore-gpu-blocklist --disable-gpu-watchdog

rem 1. Procura primeiro pelos caminhos reais do Python instalado (evita o atalho quebrado da WindowsApps)
if exist "C:\Users\Caio\AppData\Local\Python\pythoncore-3.15-64\python.exe" (
    echo Usando Python em pythoncore-3.15-64...
    "C:\Users\Caio\AppData\Local\Python\pythoncore-3.15-64\python.exe" run_gui.py
    goto check_exit
)

if exist "C:\Users\Caio\AppData\Local\Python\bin\python.exe" (
    echo Usando Python em Local\Python\bin...
    "C:\Users\Caio\AppData\Local\Python\bin\python.exe" run_gui.py
    goto check_exit
)

rem 2. Tenta o inicializador 'py'
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Usando launcher py...
    py -3 run_gui.py
    goto check_exit
)

rem 3. Tenta o python do PATH padrao
echo Tentando python do PATH...
python run_gui.py
goto check_exit

:check_exit
if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================================
    echo [AVISO] A aplicacao encerrou com o codigo de erro %ERRORLEVEL%.
    echo Se a janela fechou inesperadamente, veja a mensagem acima.
    echo ========================================================
    echo.
    pause
) else (
    echo Aplicacao finalizada normalmente.
)
