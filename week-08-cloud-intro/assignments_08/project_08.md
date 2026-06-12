# Week 8 Project — Cloud Intro & Cost Analysis

---

## Cost Analysis

### Scenario A — Lightweight Compute
**VM:** Standard_B1s (1 vCPU, 1 GB RAM), East US, Linux
**Usage:** 8 hours/day × 5 days/week ≈ 160 hours/month
**Hourly rate:** $0.0104
**Monthly total:** $1.66

### Scenario B — Heavy Analytics Workload
**VM:** Standard_NC6s_v3 (6 vCPU, 1 V100 GPU), East US, Linux — 730 hours/month (24/7)
**VM cost:** $3.06/hr × 730 hrs = $2,233.80
**Azure SQL Database:** General Purpose tier, 4 vCores ≈ $370/month
**Blob Storage:** 1 TB data ≈ $20/month
**Monthly total:** ~$2,624

---

## Write-Up

Scenario A costs only $1.66/month because the B1s is a burstable micro VM running part-time — practically free for light workloads. Scenario B was the real eye-opener: the GPU VM alone costs $2,233.80/month running 24/7, which is $26,800/year just for one machine. Running `project_08.py` produced:

```
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $1.66
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 1342.4x more than Scenario A
```

This matched the Pricing Calculator exactly. The 1,342x ratio surprised me — I initially estimated a large gap but not that large. It comes from comparing monthly costs ($2,233.80 ÷ $1.664), not hourly rates. Exploring further, I noticed that switching to a 1-year reserved instance on the NC6s_v3 drops the rate to $2.24/hr, saving over $600/month — a strong argument for commitments on long-running GPU workloads.

## Video Walkthrough (Written)

In place of a screen recording, here is a description of what the demo would show:

1. **Azure Portal** — Logged into portal.azure.com with "Code the Dream" visible in the top-right directory selector. Navigating to resource group `p200-year-ahmadavar-rg` and pointing out the storage account resource inside it.

2. **Cloud Shell** — Opening the `>_` Cloud Shell terminal. Running `ls ~/clouddrive` to show `test.txt` persists across sessions because it lives in Blob Storage, not the ephemeral shell container. Running `ls ~/.ssh` to confirm private and public key files are present.

3. **CLI commands** — Running `az group list --output table` to show all resource groups displayed as a human-readable table. Running `az account show` to review the JSON output (subscription name, tenant ID, user email).

4. **Script execution** — Pulling latest code from GitHub then running `python3 project_08.py` in Cloud Shell. Terminal prints Scenario A ($1.66/month) and Scenario B (~$2,624/month), matching the Pricing Calculator results above.
