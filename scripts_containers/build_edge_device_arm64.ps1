# Load environment variables from .env
$envFilePath = "config/.env"
if (Test-Path $envFilePath) {
    Get-Content $envFilePath | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$edgeDeviceIP = $env:EDGE_DEVICE_IP
$edgeServerIP = $env:EDGE_SERVER_IP
$dnsIp = $env:DNS_IP
$gatewayIp = $env:GATEWAY_IP
$rootPassword = $env:PASS

# Build image and push to Docker Hub
Write-Host "=== Building EdgeDevice image (arm64) ==="
docker buildx build `
  --platform linux/arm64 `
  -f app/edge_device/Dockerfile.edge_device `
  -t h3nkk44/hi-framework-edge-device:latest_arm64 `
  .

Write-Host "`n=== Image build completed ==="