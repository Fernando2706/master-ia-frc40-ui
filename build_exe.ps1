$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AppDir = $PSScriptRoot
$EntryPoint = Join-Path $AppDir "app.py"
$AppName = "FRC40_Quimicos"

Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "No existe .venv. Creando entorno virtual..."
    python -m venv .venv
}

Write-Host "Instalando dependencias..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\app\requirements.txt

Write-Host "Creando ejecutable..."
.\.venv\Scripts\python.exe -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name $AppName `
    --paths .\app\src `
    $EntryPoint

Write-Host ""
Write-Host "EXE generado en:"
Write-Host "  $(Join-Path $ProjectRoot "dist\$AppName.exe")"
