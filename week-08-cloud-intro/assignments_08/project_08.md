# Week 8 Project — Cloud Intro & Cost Analysis

Video: <paste your video link here>

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

Scenario A costs only $1.66/month because the B1s is a burstable micro VM running part-time — practically free for light workloads. Scenario B was the real eye-opener: the GPU VM alone costs $2,233/month running 24/7, which is $26,800/year just for one machine. The script output matched the calculator exactly — Scenario B's VM costs 294x more than Scenario A's full monthly bill. Exploring further, I noticed that switching to a 1-year reserved instance on the NC6s_v3 drops the rate to $2.24/hr, saving over $600/month — a strong argument for commitments on long-running GPU workloads.

Video: <paste your video link here>
