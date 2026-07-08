# Docker Container Scripts

This directory contains scripts for building and managing Docker containers.

- Update correct repository in the scripts before running them.

## Version Management

For managing images there is a version numbering system, which is used to tag the images.

- All images have tag that include architecture and version number, e.g., `edge_device_amd64_001`, `edge_device_arm64_002`.
- Running numbering is used to force download of the image instead of using the cached version.

## Scripts Overview

**Scripts for Building Docker Images:**  
`build_all_and_push.ps1` - Builds all Docker images and pushes them to the specified registry.  
`build_edge_device_amd64.ps1` - Builds image for Edge Device with AMD64 architecture.  
`build_edge_device_arm64.ps1` - Builds image for Edge Device with ARM64 architecture.  
`build_edge_server_amd64.ps1` - Builds image for Edge Server with AMD64 architecture.  

**Scripts for Managing Docker Images:**  
`push_containers.ps1` - Pushes selected Docker images to the Docker Hub.  
`tag_containers.ps1` - Tags Docker images for the specified architecture and version number.  

**Scripts for Running Docker Containers:**  
`start_local_containers_amd64.ps1` - Starts local Docker containers for AMD64 architecture.  
`start_local_containers_arm64.ps1` - Starts local Docker containers for ARM64 architecture.  

## Current Workflow

The current repo workflow is thesis reproduction on ExPECA. Use the runbook in:

```text
docs/thesis_reproduction.md
```
