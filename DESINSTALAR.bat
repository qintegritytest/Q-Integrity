@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
echo.
echo ========================================
echo    DESINSTALADOR QINTEGRITY
echo ========================================
echo.
echo Este proceso eliminará el entorno virtual
echo y las dependencias instaladas.
echo.
echo Sus archivos de proyecto se mantienen intactos.
echo.

set /p CONFIRM="¿Desea continuar? (S/N): "
if /i not "%CONFIRM%"=="S" (
    echo.
    echo Desinstalación cancelada.
    pause
    exit /b 0
)

echo.
echo Eliminando entorno virtual...
if exist venv (
    rmdir /s /q venv
    if errorlevel 1 (
        echo.
        echo ADVERTENCIA: No se pudo eliminar completamente la carpeta venv
        echo Intente eliminarla manualmente desde el Explorador de archivos
    ) else (
        echo   ✓ Entorno virtual eliminado
    )
) else (
    echo   ℹ No se encontró carpeta venv
)

echo.
echo Eliminando archivos de Python en caché...
if exist __pycache__ (
    rmdir /s /q __pycache__
)

REM Buscar and eliminar otras carpetas __pycache__
for /d /r . %%i in (__pycache__) do (
    rmdir /s /q "%%i" 2>nul
)

echo   ✓ Archivos de caché eliminados

echo.
echo ========================================
echo    ✓ DESINSTALACIÓN COMPLETADA
echo ========================================
echo.
echo Si desea reinstalar, ejecute INSTALADOR.bat
echo.
pause
