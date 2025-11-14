# Configuration
$edgeDeviceImage = "h3nkk44/hi-framework-edge-device:latest_arm64_013"
$edgeServerImage = "h3nkk44/hi-framework-edge-server:latest_amd64_013"
$edgeDeviceContainer = "edge_device"
$edgeServerContainer = "edge_server"
$edgeDevicePort = 8000
$edgeServerPort = 8001
$sshPortDevice = 2222
$sshPortServer = 2223
$networkName = "edge_net"

# Load environment variables from .env
$envFilePath = "config/.env"
if (Test-Path $envFilePath) {
    Get-Content $envFilePath | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

$edgeDeviceIP = $env:EDGE_DEVICE_IP
$edgeServerIP = $env:EDGE_SERVER_IP
$dnsIp = $env:DNS_IP
$gatewayIp = $env:GATEWAY_IP
$rootPassword = $env:PASS

# Create Docker network if it doesn't exist
if (-not (docker network ls --format '{{.Name}}' | Select-String -Pattern $networkName)) {
    Write-Host "=== Creating Docker network '$networkName' ==="
    docker network create $networkName
} else {
    Write-Host "=== Docker network '$networkName' already exists ==="
}

# Start EdgeServer container
Write-Host "=== Starting EdgeServer container ==="
docker run -d --network $networkName `
  --cap-add=NET_ADMIN `
  -p "$edgeServerPort`:8001" `
  -p "$sshPortServer`:22" `
  --name $edgeServerContainer `
  -e EDGE_DEVICE_IP=$edgeDeviceIP `
  -e EDGE_SERVER_IP=$edgeServerIP `
  -e DNS_IP=$dnsIp `
  -e GATEWAY_IP=$gatewayIp `
  -e PASS=$rootPassword `
  $edgeServerImage

# Start EdgeDevice container
Write-Host "=== Starting EdgeDevice container ==="
docker run -d --platform linux/arm64 --network $networkName `
  --cap-add=NET_ADMIN `
  -p "$edgeDevicePort`:8000" `
  -p "$sshPortDevice`:22" `
  --name $edgeDeviceContainer `
  -e EDGE_DEVICE_IP=$edgeDeviceIP `
  -e EDGE_SERVER_IP=$edgeServerIP `
  -e DNS_IP=$dnsIp `
  -e GATEWAY_IP=$gatewayIp `
  -e PASS=$rootPassword `
  $edgeDeviceImage

# Check running containers
Write-Host "=== Containers running ==="
docker ps --filter "name=$edgeDeviceContainer"
docker ps --filter "name=$edgeServerContainer"

Write-Host "`n=== EdgeDevice and EdgeServer containers are up and running ==="
