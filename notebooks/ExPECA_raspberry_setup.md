# Raspberry Pi 4 Worker Node Setup Guide

*Ubuntu Server 22.04.5 LTS*

---

## 1. Flash OS to SD Card

### Download Raspberry Pi Imager
- Get it from: [https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)

### Prepare the SD Card
- Insert SD card into your PC.
- Format it if necessary using your OS's disk utility.

### Write OS Image
- Launch Raspberry Pi Imager.
- Use these settings:
  - **OS**: `Ubuntu Server 22.04.5 LTS (Raspberry Pi 4)`
  - **Hostname**: `edgedevice.local`
  - **Username**: `expeca`
  - **Password**: `expeca`
  - **SSH**: Enabled (password auth)

---

## 2. Boot and Connect to Raspberry Pi

- Insert SD card, power on the Pi.
- Wait for initial boot (a few minutes).
- Connect via SSH:

```bash
ssh expeca@edgedevice.local
```

---

## 3. System Update

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 4. User Setup

### Add `expeca` User (if not preconfigured)

```bash
sudo useradd -m expeca
echo 'expeca:expeca' | sudo chpasswd
sudo usermod -aG sudo expeca
```

### Grant Passwordless Sudo

```bash
sudo visudo
```

Add at the bottom:

```bash
expeca ALL=(ALL) NOPASSWD: ALL
```

### Add SSH Key Access
- Append the controller's public key to:

```bash
/home/expeca/.ssh/authorized_keys
```

---

## 5. Install Dependencies

### Docker 20.10.22

```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt install docker-ce=5:20.10.22~3-0~ubuntu-focal
sudo apt install docker-ce-cli=5:20.10.22~3-0~ubuntu-focal
sudo apt install docker-ce-rootless-extras=5:20.10.22~3-0~ubuntu-focal
```

Verify:

```bash
sudo docker version
```

Expected version: `20.10.22`

### Python & Tools

```bash
sudo apt install -f -y python3 python3-venv python3-pip jq git
```

### Docker SDK for Python

```bash
sudo apt install python3-requests python3-urllib3 python3-docker
```

---

## 6. Install Go (v1.19.2, ARM64)

```bash
wget https://dl.google.com/go/go1.19.2.linux-arm64.tar.gz
sudo tar -xvf go1.19.2.linux-arm64.tar.gz
sudo mv go /usr/local
```

> If `/usr/local/go` exists, delete it first.

### Set Environment Variables

Edit `.bashrc`:

```bash
vim ~/.bashrc
```

Add:

```bash
GOROOT=/usr/local/go
GOPATH=$HOME/Projects/Proj1
PATH=$GOPATH/bin:$GOROOT/bin:$PATH
```

Apply changes:

```bash
su - ${USER}
```

Check version:

```bash
go version
```

---

## 7. Disable iSCSI Services (if installed)

```bash
sudo systemctl stop open-iscsi iscsid iscsid.socket iscsiuio.socket
sudo systemctl disable iscsid.socket iscsiuio.socket iscsid.service
sudo apt remove --purge open-iscsi
```

---

## 8. Mount Kernel Configs at Boot

### Edit `/etc/rc.local`

```bash
sudo nano /etc/rc.local
```

Add:

```bash
#!/bin/bash
mount -t configfs none /sys/kernel/config
exit 0
```

Make it executable:

```bash
sudo chmod +x /etc/rc.local
```

---

## 9. SSH Configuration

Edit:

```bash
sudo nano /etc/ssh/sshd_config
```

Ensure:

```
PasswordAuthentication yes
PubkeyAuthentication no
```

Restart:

```bash
sudo systemctl restart ssh
```

---

## 10. Configure Networking with Netplan

### Check Interface Names

```bash
ip link show
```

### Edit Netplan Config

```bash
sudo nano /etc/netplan/01-netcfg.yaml
```

Example:

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no

    enxe01cfcdd1677:
      dhcp4: no
      addresses: [10.10.2.201/16]

    enxe01cfcdd1694:
      dhcp4: no
      addresses: [10.20.111.201/24]
```

Apply:

```bash
sudo netplan apply
```

If warned:

```bash
sudo chmod 600 /etc/netplan/01-netcfg.yaml
sudo netplan apply
```

Verify:

```bash
ip a
```

---

## Interface Roles

- `enxe01cfcdd1677`: Management (10.10.2.201/16)  
- `enxe01cfcdd1694`: Internal (10.20.111.201/24)  
- `eth0`: Tenant (no static IP)

---
