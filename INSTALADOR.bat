
@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM Verificar permisos de administrador
net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo    ERROR: Permisos Insuficientes
    echo ========================================
    echo.
    echo Este instalador requiere permisos de administrador.
    echo.
    echo Por favor:
    echo 1. Haga clic derecho en INSTALADOR.bat
    echo 2. Seleccione "Ejecutar como administrador"
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    INSTALADOR QINTEGRITY
echo ========================================
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Verificar si Python está instalado
echo [1/5] Verificando Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python no está instalado o no está en PATH
    echo Descargue Python desde https://www.python.org/downloads/
    echo Asegúrese de marcar "Add Python to PATH" durante la instalación
    echo.
    pause
    exit /b 1
)

REM Obtener versión de Python
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   ✓ %PYTHON_VERSION% encontrado

REM Crear entorno virtual
echo.
echo [2/5] Creando entorno virtual...
if exist venv (
    echo   ✓ Entorno virtual ya existe
) else (
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
    echo   ✓ Entorno virtual creado
)

REM Activar entorno virtual
echo.
echo [3/5] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: No se pudo activar el entorno virtual
    pause
    exit /b 1
)
echo   ✓ Entorno virtual activado

REM Instalar dependencias de Python
echo.
echo [4/5] Instalando dependencias de Python...
echo   (esto puede tardar unos minutos)
python -m pip install --upgrade pip > nul 2>&1

REM Verificar que requirements.txt existe
if not exist requirements.txt (
    echo.
    echo ERROR: No se encontró 'requirements.txt'
    echo Asegúrese de estar en la carpeta correcta del proyecto
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron instalar las dependencias de Python
    pause
    exit /b 1
)
echo   ✓ Dependencias de Python instaladas

REM Instalar Tesseract OCR
echo.
echo [5/5] Instalando Tesseract OCR...
where tesseract > nul 2>&1
if errorlevel 1 (
    echo   Tesseract no detectado. Instalando...
    
    REM Verificar si Chocolatey está instalado
    where choco > nul 2>&1
    if errorlevel 1 (
        echo   Instalando Chocolatey (gestor de paquetes)...
        powershell -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && set PATH="%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
        
        if errorlevel 1 (
            echo.
            echo ERROR: No se pudo instalar Chocolatey
            echo Por favor, intente descargar Tesseract OCR manualmente desde:
            echo https://github.com/UB-Mannheim/tesseract/wiki
            echo.
            pause
            exit /b 1
        )
        echo   ✓ Chocolatey instalado correctamente
    ) else (
        echo   ✓ Chocolatey ya está instalado
    )
    
    REM Instalar Tesseract con Chocolatey
    echo   Instalando Tesseract OCR (esto puede tardar unos minutos)...
    choco install -y tesseract
    
    if errorlevel 1 (
        echo.
        echo ERROR: No se pudo instalar Tesseract OCR con Chocolatey
        echo Por favor, intente descargar manualmente desde:
        echo https://github.com/UB-Mannheim/tesseract/wiki
        echo.
        pause
        exit /b 1
    )
    
    REM Configurar la variable de entorno para pytesseract
    set TESSERACT_PATH=C:\Program Files\Tesseract-OCR
    if exist "!TESSERACT_PATH!\tesseract.exe" (
        setx TESSERACT_CMD "!TESSERACT_PATH!\tesseract.exe"
        echo   ✓ Tesseract OCR instalado en: !TESSERACT_PATH!
    ) else (
        echo.
        echo ADVERTENCIA: No se encontró Tesseract en la ruta esperada
        echo Se intentará utilizar la instalación por defecto
    )
) else (
    echo   ✓ Tesseract OCR ya está instalado
)

REM Crear archivo ejecutor
echo.
echo [✓] Creando script de ejecución...
(
    echo @echo off
    echo chcp 65001 ^> nul
    echo cd /d "%%~dp0"
    echo call venv\Scripts\activate.bat
    echo python -m streamlit run app.py
    echo pause
) > EJECUTAR.bat
echo   ✓ Archivo EJECUTAR.bat creado

REM Crear acceso directo (opcional en Windows)
echo.
echo ========================================
echo    ✓ INSTALACIÓN COMPLETADA
echo ========================================
echo.
echo Para ejecutar la aplicación:
echo   1. Haga doble clic en EJECUTAR.bat
echo   O en terminal:
echo   2. Ejecute: EJECUTAR.bat
echo.
echo La aplicación se abrirá en su navegador predeterminado.
echo.
pause
