param([switch]$CPUOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
cmake -S . -B build -A x64
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed. Install Visual Studio 2022 Desktop development with C++ and CMake.' }
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
ctest --test-dir build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
$demoArgs = @('--demo', '--tiles', '8', '--file', 'dm3d_1GB.vmem')
if ($CPUOnly) { $demoArgs += '--cpu' }
& .\build\Release\DrMoagi3D.exe @demoArgs | Tee-Object -FilePath verification.log
if ($LASTEXITCODE -ne 0) { throw 'End-to-end verification failed' }
