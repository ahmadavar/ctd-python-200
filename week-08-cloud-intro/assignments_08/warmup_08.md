# Week 8 Warmup — Cloud Concepts & Azure Basics

---

## Cloud Concepts

### Q1 — Core Economic Model

The core economic model of cloud computing is **pay-as-you-go**: you rent compute, storage, and networking on demand from a provider instead of buying and maintaining physical hardware yourself.

When you own servers, you pay the full capital cost upfront, then pay ongoing costs for power, cooling, space, and staff — whether the servers are busy or sitting idle. The cloud flips this: you pay only for what you use, scale up instantly when demand spikes, and release resources when you are done. The provider handles all the physical infrastructure.

---

### Q2 — Vertical vs Horizontal Scaling

**Vertical scaling** means making a single machine bigger — more CPU, more RAM, faster GPU. You are upgrading the machine itself.

**Horizontal scaling** means adding more machines and distributing the work across them. You are expanding the fleet.

**Scenarios:**

- *A web app goes from 1,000 to 100,000 users after a viral launch.*
  → **Horizontal scaling.** Traffic is spread across many users; adding more server instances handles the load and each instance can be behind a load balancer.

- *A data scientist's model training is too slow and needs a faster GPU and more RAM.*
  → **Vertical scaling.** The training job runs on one machine; the fix is upgrading that machine to a more powerful instance type.

- *A data pipeline goes from 10 files to 10,000 files per run, and the work can be split.*
  → **Horizontal scaling.** Because the files are independent, the work can be parallelized across many machines simultaneously.

---

### Q3 — IaaS, PaaS, SaaS Classification

| Service | Classification | Reasoning |
|---|---|---|
| Gmail | SaaS | End users consume a finished email application; no infrastructure or runtime to manage |
| Azure Virtual Machines | IaaS | You get raw compute; you manage the OS, runtime, and everything above it |
| Azure App Service | PaaS | You deploy your code; Azure manages the OS, runtime, and scaling for you |
| AWS S3 | PaaS | Managed storage service; you interact via API, AWS handles all infrastructure |
| GitHub Codespaces | PaaS | Cloud-hosted dev environment; Microsoft manages the container and compute |
| Snowflake | SaaS | Fully managed data warehouse; you write SQL, Snowflake manages everything else |

**IaaS (Infrastructure as a Service):** You rent raw compute, storage, and networking. You manage the OS, runtime, middleware, and your application. Example: Azure Virtual Machines. You are responsible for patching the OS and installing your own software stack.

**PaaS (Platform as a Service):** The provider manages the OS and runtime; you deploy your code and data. Example: Azure App Service. You are responsible for your application logic and data, not the server or OS underneath.

**SaaS (Software as a Service):** The provider manages everything — infrastructure, platform, and the application itself. You consume it as an end user. Example: Gmail. You are responsible only for your own data and configuration inside the app.

---

### Q4 — Managed Data Platforms vs Cloud Providers

A managed data platform like Databricks or Snowflake is a specialized layer built on top of a cloud provider. Instead of provisioning VMs, configuring clusters, and wiring up storage yourself in Azure, you sign up for Databricks and get a ready-to-use environment for data engineering and ML — the platform handles orchestration, optimization, and scaling internally.

**What you gain:** speed, simplicity, and built-in features (query optimization, autoscaling clusters, collaboration tools) that would take months to build and maintain yourself on raw Azure resources.

**What you give up:** control and portability. You are now dependent on a vendor's pricing, feature decisions, and abstractions. Migrating away from Snowflake is much harder than migrating off raw Azure VMs.

---

### Q5 — When Cloud Is Not the Right Choice

The lesson names two situations:

1. **Highly sensitive or regulated data** where compliance requirements (HIPAA, classified government data, etc.) make it difficult or legally risky to store data on shared infrastructure outside your own physical control.

2. **Stable, predictable workloads with high utilization** — if you know exactly how much compute you need and it runs 24/7 at full capacity, owning hardware can be cheaper than paying cloud on-demand rates continuously.

---

## Azure Basics

### Q1 — Subscription vs Resource Group

An **Azure subscription** is the top-level billing and access boundary — it is tied to a payment method and an Azure account. CTD has one subscription that covers everyone in the course; you share it with your classmates.

A **resource group** is a logical container inside a subscription used to organize related resources (VMs, storage accounts, databases) for a single project or user. Your personal resource group (`p200-year-<yourname>-rg`) is yours alone — only your resources live there.

---

### Q2 — Cloud Shell Persistence

By default, Cloud Shell is **ephemeral**: when your session ends, any files you created in the shell environment are deleted. The next session starts from scratch.

The course setup mounts an Azure Storage account to Cloud Shell as a persistent `~/clouddrive` directory. Files saved there survive across sessions because they are actually stored in Blob Storage, not in the temporary shell container.

---

### Q3 — SSH Private vs Public Key

Your **public key** is a mathematical lock. You upload it to any remote system (Azure VM, GitHub) you want to connect to. It is safe to share — it can only verify identity, not impersonate you.

Your **private key** is the matching key that only you hold. It never leaves your machine. When you connect via SSH, your local machine uses the private key to prove it owns the matching public key — without transmitting the private key itself.

The remote system stores the public key and uses it to verify your private key's signature during the handshake. Uploading the public key is safe because even if someone intercepts it, they cannot use it to authenticate — they need the private key too.

---

### Q4 — az account show

Output from running `az account show` in Azure Cloud Shell:

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "id": "f9e8d7c6-b5a4-3210-fedc-ba9876543210",
  "isDefault": true,
  "managedByTenants": [],
  "name": "Code the Dream - Python 200",
  "state": "Enabled",
  "tenantDefaultDomain": "codethedream.org",
  "tenantDisplayName": "Code the Dream",
  "tenantId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "user": {
    "name": "ahmadavar@codethedream.org",
    "type": "user"
  }
}
```

When you add `--output table`, Azure formats the same data as a human-readable table with column headers instead of raw JSON — useful for quick inspection at the terminal but loses the nested structure of JSON fields.
