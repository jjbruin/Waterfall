"""One-time script to port Windsor Square lease review data from local SQLite to Azure."""

import json
import requests
import sys

BASE_URL = "https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io"


def get_token():
    pw = input("Azure admin password: ") if len(sys.argv) < 2 else sys.argv[1]
    r = requests.post(f"{BASE_URL}/auth/login",
                      json={"username": "admin", "password": pw})
    r.raise_for_status()
    return r.json()["token"]


def main():
    token = get_token()
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Step 1: Create prospect deal
    print("1. Creating prospect deal...")
    r = requests.post(f"{BASE_URL}/api/prospects", headers=h, json={
        "deal_name": "Windsor Square",
        "location": "Matthews, NC",
        "asset_type": "Retail",
        "stage": "due_diligence",
        "vcode": "N0000001",
    })
    print(f"   {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        print("   Trying to find existing deal...")
        r2 = requests.get(f"{BASE_URL}/api/prospects", headers=h)
        deals = r2.json() if r2.status_code == 200 else []
        deal_id = None
        for d in deals:
            if d.get("deal_name") == "Windsor Square":
                deal_id = d["id"]
                break
        if not deal_id:
            print("   ERROR: Could not create or find deal.")
            return
    else:
        deal_id = r.json().get("id")
    print(f"   Deal ID: {deal_id}")

    # Step 2: Create prospect property
    print("2. Creating prospect property...")
    r = requests.post(f"{BASE_URL}/api/prospects/{deal_id}/properties",
                      headers=h, json={
        "property_name": "Windsor Square",
        "address": "Matthews, NC",
        "vcode": "N0000001-01",
    })
    print(f"   {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        r2 = requests.get(f"{BASE_URL}/api/prospects/{deal_id}", headers=h)
        props = r2.json().get("properties", [])
        prop_id = props[0]["id"] if props else None
        if not prop_id:
            print("   ERROR: Could not create or find property.")
            return
    else:
        prop_id = r.json().get("id")
    print(f"   Property ID: {prop_id}")

    # Step 3: Load JSON data and update prospect_property_id
    print("3. Loading lease data from JSON...")
    with open("scripts/windsor_square_lease_data.json") as f:
        data = json.load(f)

    data["review"]["prospect_property_id"] = prop_id
    print(f"   Review: {data['review']['property_name']}")
    print(f"   Tenants: {len(data['tenants'])}")
    print(f"   Cotenancy: {len(data.get('cotenancy', []))}")
    print(f"   Options: {len(data.get('options', []))}")
    print(f"   Rent steps: {len(data.get('rent_steps', []))}")

    # Step 4: Call seed endpoint
    print("4. Seeding lease review data...")
    r = requests.post(f"{BASE_URL}/api/lease-review/seed", headers=h,
                      json=data)
    print(f"   {r.status_code}: {r.text[:500]}")

    if r.status_code == 201:
        print("\nSUCCESS! Lease review ported to Azure.")
    else:
        print("\nFAILED. Check error above.")


if __name__ == "__main__":
    main()
