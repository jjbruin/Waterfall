# VPN Tunnel to MRI — FULLY OPERATIONAL (Jun 25, 2026)

## Status: FULLY WORKING — both PMX and IM SQL servers accessible from Azure

## MRI SQL Servers (via VPN tunnel)
| Server | Tunnel IP | Instance | Database | Purpose |
|---|---|---|---|---|
| PMX | 10.219.226.9:1433 | NCHDBS184\P22 | BV6899900001 | Accounting, Relationships, Commitments |
| IM | 10.219.226.10:1433 | GCHDBS005\P5 | PSC | ISBS, Valuations, Loans, Occupancy, Tenants |

- **Credentials**: UID=PSCVPN, PWD=NVc8MkB^PlRuv*
- **Config**: `flask_app/services/mri_service.py` (committed as `2d8d1ec`)
- **Old FortiClient IPs**: .17 (PMX) and .18 (IM) — no longer used; tunnel IPs are .9 and .10

## Working VPN Configuration

### Phase 1 (IKE SA)
| Parameter | Value |
|---|---|
| Encryption | AES-256 |
| Integrity | SHA-256 |
| DH Group | **14** (2048-bit MODP) |
| Lifetime | 28800s (8 hours) |
| Authentication | Pre-Shared Key |

### Phase 2 (IPsec SA / Quick Mode)
| Parameter | Value |
|---|---|
| Encryption | AES-256 |
| Integrity | SHA-256 |
| PFS Group | 14 |
| Lifetime | 3600s (1 hour) |
| Traffic Selectors | Local 10.0.0.0/16 <-> Remote 10.219.226.8/29 |

### Endpoints
| Side | Public IP | Local Subnet |
|---|---|---|
| Azure | 48.194.101.189 | 10.0.0.0/16 |
| MRI | 64.37.251.115 | 10.219.226.8/29 |

### PSK
`QVYGlQSlBrn2zwv$A*4F6$1W#CverF38`

## Azure Resource Summary
| Resource | Name | Detail |
|---|---|---|
| VPN Gateway | vpngw-waterfall-dev | VpnGw1AZ, Generation1, RouteBased |
| Gateway Public IP | pip-vpngw-waterfall | 48.194.101.189 (static, Standard SKU) |
| Local Network Gateway | lgw-mri | Peer: 64.37.251.115, Remote: 10.219.226.8/29 |
| Connection | vpn-to-mri | IKEv2, PSK, Default mode, DPD 45s, DHGroup14 policy |
| VNet | vnet-waterfall-dev | 10.0.0.0/16 |
| GatewaySubnet | GatewaySubnet | 10.0.2.0/27 (no NSG, no route table, no NAT) |
| Container App Subnet | snet-containerapp | 10.0.0.0/23, route table: rt-containerapp |
| Route Table | rt-containerapp | 10.219.226.8/29 -> VirtualNetworkGateway |
| NAT Gateway | nat-waterfall-dev | 20.127.96.240, on snet-containerapp only |
| Diagnostic Logs | vpn-ike-diag | IKE/Tunnel/Gateway/Route logs -> stwaterfalldev |
| Container App | app-waterfall-dev-v2 | Revision v39, deployed Jun 24 |

## Key Lessons Learned

### DH Group Issue (Root Cause of Jun 23-24 failures)
- Azure VPN Gateway Gen1 (VpnGw1AZ) **silently ignores ECP384 (DH21)** and sends DhGroup20 instead
- Config is accepted without error but the IKE daemon uses DhGroup20 in SA_INIT proposals
- **DHGroup14 IS respected** — confirmed via IKE diagnostic logs
- MRI's FortiGate was configured for DH21 only -> `NO_PROPOSAL_CHOSEN` on every attempt
- **Fix**: MRI added DH14 to accepted proposals -> tunnel came up immediately

### Azure CLI Gotchas
- **`az network vpn-connection update` wipes PSK** — always re-set PSK after any connection update
- **`ipsec-policy clear` can also wipe PSK** — same issue
- **`ipsec-policy add` preserves PSK** — but verify anyway
- **Gateway reset preserves PSK** — confirmed
- **`egressBytesTransferred: 0` is misleading** — failed IKE attempts don't increment the counter; use diagnostic logs instead
- **`MSYS_NO_PATHCONV=1`** needed for Git Bash to prevent path mangling on resource IDs

### Diagnostic Logging
- Enabled: IKEDiagnosticLog, TunnelDiagnosticLog, GatewayDiagnosticLog, RouteDiagnosticLog
- Storage account: `stwaterfalldev`
- Containers: `insights-logs-ikediagnosticlog`, `insights-logs-tunneldiagnosticlog`, etc.
- Download logs: `az storage blob download --account-name stwaterfalldev --container-name insights-logs-ikediagnosticlog --name "..." --auth-mode key -f output.json`

## Testing Commands
```bash
# Tunnel status
MSYS_NO_PATHCONV=1 az network vpn-connection show -g rg-waterfall-dev -n vpn-to-mri --query "{status:connectionStatus, egress:egressBytesTransferred, ingress:ingressBytesTransferred}" -o json

# IKE SAs
MSYS_NO_PATHCONV=1 az network vpn-connection list-ike-sas -g rg-waterfall-dev -n vpn-to-mri -o json

# SQL test from container (PMX)
MSYS_NO_PATHCONV=1 az containerapp exec -g rg-waterfall-dev -n app-waterfall-dev-v2 --command "python -c \"import pyodbc; conn=pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=10.219.226.9,1433;DATABASE=BV6899900001;UID=PSCVPN;PWD=NVc8MkB^PlRuv*;TrustServerCertificate=yes;'); cursor=conn.cursor(); cursor.execute('SELECT @@SERVERNAME'); print(cursor.fetchone()[0]); conn.close()\""

# SQL test from container (IM)
MSYS_NO_PATHCONV=1 az containerapp exec -g rg-waterfall-dev -n app-waterfall-dev-v2 --command "python -c \"import pyodbc; conn=pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=10.219.226.10,1433;DATABASE=PSC;UID=PSCVPN;PWD=NVc8MkB^PlRuv*;TrustServerCertificate=yes;'); cursor=conn.cursor(); cursor.execute('SELECT @@SERVERNAME'); print(cursor.fetchone()[0]); conn.close()\""
```

## Rebuild Steps (if gateway needs to be recreated)
```bash
# 1. Delete connection then gateway
az network vpn-connection delete -g rg-waterfall-dev -n vpn-to-mri
az network vnet-gateway delete -g rg-waterfall-dev -n vpngw-waterfall-dev  # ~10 min

# 2. Create gateway (~30-45 min)
az network vnet-gateway create -g rg-waterfall-dev -n vpngw-waterfall-dev \
  --vnet vnet-waterfall-dev --public-ip-addresses pip-vpngw-waterfall \
  --gateway-type Vpn --vpn-type RouteBased --sku VpnGw1AZ --no-wait

# 3. Create connection
az network vpn-connection create -g rg-waterfall-dev -n vpn-to-mri \
  --vnet-gateway1 vpngw-waterfall-dev --local-gateway2 lgw-mri \
  --shared-key 'QVYGlQSlBrn2zwv$A*4F6$1W#CverF38' --location eastus

# 4. Apply IPsec policy (use DHGroup14, NOT ECP384)
az network vpn-connection ipsec-policy add -g rg-waterfall-dev --connection-name vpn-to-mri \
  --ike-encryption AES256 --ike-integrity SHA256 --dh-group DHGroup14 \
  --ipsec-encryption AES256 --ipsec-integrity SHA256 --pfs-group PFS2048 \
  --sa-lifetime 3600 --sa-max-size 102400000

# 5. Re-set PSK (policy add may wipe it)
az network vpn-connection shared-key update -g rg-waterfall-dev \
  --connection-name vpn-to-mri --value 'QVYGlQSlBrn2zwv$A*4F6$1W#CverF38'

# 6. Re-enable diagnostics
az monitor diagnostic-settings create -n vpn-ike-diag \
  --resource "/subscriptions/ba0c885f-7fce-40a7-9bcf-2aaf81118716/resourceGroups/rg-waterfall-dev/providers/Microsoft.Network/virtualNetworkGateways/vpngw-waterfall-dev" \
  --storage-account stwaterfalldev \
  --logs '[{"category":"IKEDiagnosticLog","enabled":true},{"category":"TunnelDiagnosticLog","enabled":true},{"category":"GatewayDiagnosticLog","enabled":true},{"category":"RouteDiagnosticLog","enabled":true}]'
```

## Cleanup Completed
- Test VM, NIC, public IP, NSG, OS disk, snet-vm subnet — all deleted
- Temp IKE log files — deleted

## Timeline
- **Jun 23**: Initial tunnel attempts — 0 egress, multiple resets, PSK wipe discovered
- **Jun 24**: Diagnostic logs -> DH group mismatch found -> DHGroup14 fix -> tunnel up -> MRI firewall/routing fixed -> PMX working -> IM port 1433 opened -> **both servers fully operational**
- **Jun 24**: mri_service.py updated (.9/.10), deployed as v39
- **Jun 25**: COA query fix — `vCOA` view uses `vaccount` not `vcode`; all 13 MRI queries now working from Azure. Deployed as v41.
