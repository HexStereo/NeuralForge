$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "NeuralForge.lnk"

$BaseDir = Split-Path -Parent $PSScriptRoot
$PythonwPath = Join-Path $BaseDir ".venv\Scripts\pythonw.exe"
$AppPyPath = Join-Path $BaseDir "desktop_app.py"
$IcoPath = Join-Path $PSScriptRoot "neuralforge.ico"

if (Test-Path $PythonwPath) {
    $TargetExe = $PythonwPath
    $Arguments = "`"$AppPyPath`""
} else {
    $TargetExe = Join-Path $PSScriptRoot "start_neuralforge.bat"
    $Arguments = ""
}

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetExe
if ($Arguments -ne "") {
    $Shortcut.Arguments = $Arguments
}
$Shortcut.WorkingDirectory = $BaseDir
$Shortcut.IconLocation = "$IcoPath,0"
$Shortcut.Description = "NeuralForge - AI Research Studio"
$Shortcut.Save()

Write-Host "[SUCCESS] NeuralForge Desktop Shortcut updated at: $ShortcutPath"
Write-Host "  Target: $TargetExe $Arguments"
Write-Host "  Icon:   $IcoPath"
