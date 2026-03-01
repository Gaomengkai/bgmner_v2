param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

function Resolve-PythonExe {
    if ($env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return "python"
}

$pythonExe = Resolve-PythonExe

Write-Host "[bgmner-bert] Checking environment..."

$isWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows
)
if (-not $isWindows) {
    Fail "This script only supports Windows."
}

$pythonVersion = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    Fail "Python is not available. Resolved path: $pythonExe"
}
if ($pythonVersion.Trim() -ne "3.12") {
    Fail "Python 3.12 is required. Current version: $($pythonVersion.Trim())"
}

$gpuNames = @()
try {
    $gpuNames = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
} catch {
    Fail "Unable to query GPU information from Win32_VideoController."
}

if (-not ($gpuNames | Where-Object { $_ -match "AMD|Radeon" })) {
    Fail "No AMD/Radeon GPU detected. Current adapters: $($gpuNames -join '; ')"
}

if ($Force) {
    Write-Host "[bgmner-bert] Removing existing torch/rocm packages..."
    & $pythonExe -m pip uninstall -y torch torchaudio torchvision rocm rocm_sdk_core rocm_sdk_devel rocm_sdk_libraries_custom
}

Write-Host "[bgmner-bert] Upgrading pip..."
& $pythonExe -m pip install --upgrade pip

Write-Host "[bgmner-bert] Installing ROCm SDK packages..."
& $pythonExe -m pip install `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

Write-Host "[bgmner-bert] Installing PyTorch ROCm packages..."
& $pythonExe -m pip install `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl

Write-Host "[bgmner-bert] Verifying torch installation..."
& $pythonExe -c "import torch; print('torch=', torch.__version__); print('hip=', getattr(torch.version, 'hip', None)); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "[bgmner-bert] Done."
