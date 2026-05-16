# Infrastructure Notes

This project is designed to move through the following layers:

1. Local validation on your Windows machine
2. Ubuntu VM execution
3. Proxmox-hosted VM execution
4. Secure access with Tailscale
5. Containerized deployment with Dokploy

## Recommended VM layout

- Proxmox VE host on separate hardware if possible
- One Ubuntu VM for the benchmark workload
- Optional second VM for Dokploy to avoid noisy benchmark timings

## Why Dokploy is not part of the first benchmark

Dokploy is useful for packaging and repeatable deployment, but it is not the right place to debug the first quantum simulation. Run the benchmark inside the VM first. Once it is stable, deploy the same Docker image with Dokploy.

## Suggested Ubuntu VM size

- 4 to 8 vCPU
- 8 to 16 GB RAM
- 30 GB disk

## Tailscale placement

Install Tailscale on the Ubuntu VM first. Add it to the Proxmox host later if you want secure remote administration of the virtualization layer.

## Benchmarking advice

If you want clean performance numbers, avoid running Dokploy workloads on the same VM as the benchmark. Shared CPU and memory pressure will skew your timings.

