param(
    [string]$Url = "http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip",
    [string]$OutputDir = "data/raw/JAAD",
    [switch]$KeepZip,
    [switch]$UseExistingZip
)

# Windows PowerShell script for downloading and extracting JAAD video clips.
# Default flow:
# 1. Create data/raw/JAAD
# 2. Download JAAD_clips.zip, or use an existing zip with -UseExistingZip
# 3. Extract the zip file
# 4. Remove the downloaded zip file unless -KeepZip is used

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ResolvedOutputDir = Join-Path $ProjectRoot $OutputDir
$ZipPath = Join-Path $ResolvedOutputDir "JAAD_clips.zip"
$TempZipPath = Join-Path $ResolvedOutputDir ("JAAD_clips.{0}.download.zip" -f $PID)

Write-Host "Project root: $ProjectRoot"
Write-Host "Output directory: $ResolvedOutputDir"

if (-not (Test-Path $ResolvedOutputDir)) {
    New-Item -ItemType Directory -Path $ResolvedOutputDir | Out-Null
}

Write-Host "Starting JAAD clips download..."
Write-Host "URL: $Url"

if ($UseExistingZip) {
    if (-not (Test-Path $ZipPath)) {
        throw "UseExistingZip was set, but the zip file does not exist: $ZipPath"
    }
    Write-Host "Using existing zip: $ZipPath"
    $ArchivePath = $ZipPath
}
else {
    Write-Host "Temporary zip: $TempZipPath"
    Invoke-WebRequest -Uri $Url -OutFile $TempZipPath
    $ArchivePath = $TempZipPath
}

Write-Host "Extracting archive..."
Expand-Archive -Path $ArchivePath -DestinationPath $ResolvedOutputDir -Force

if ($UseExistingZip) {
    Write-Host "Existing zip was used. Keeping zip file: $ZipPath"
}
elseif ($KeepZip) {
    Write-Host "KeepZip was set. Saving zip file: $ZipPath"
    if (Test-Path $ZipPath) {
        try {
            Remove-Item -LiteralPath $ZipPath -Force
        }
        catch {
            $FallbackZipPath = Join-Path $ResolvedOutputDir ("JAAD_clips.{0}.zip" -f (Get-Date -Format "yyyyMMddHHmmss"))
            Write-Host "Existing zip is locked. Keeping the downloaded zip at: $FallbackZipPath"
            Move-Item -LiteralPath $TempZipPath -Destination $FallbackZipPath -Force
            Write-Host "Done."
            exit 0
        }
    }
    Move-Item -LiteralPath $TempZipPath -Destination $ZipPath -Force
}
else {
    Write-Host "Removing temporary zip file: $TempZipPath"
    Remove-Item -LiteralPath $TempZipPath -Force
}

Write-Host "Done."
