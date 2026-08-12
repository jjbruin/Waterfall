"""Populate cotenancy references and detailed clause analysis for Windsor Square."""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['FLASK_ENV'] = 'development'

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app import create_app
app = create_app()

with app.app_context():
    from flask_app.db import get_engine
    from sqlalchemy import text
    engine = get_engine()

    # Read all cotenancy clauses and extract named tenants
    with engine.connect() as conn:
        clauses = conn.execute(text("""
            SELECT c.id, c.tenant_id, t.tenant_name, c.clause_text
            FROM lease_cotenancy c
            JOIN lease_tenants t ON t.id = c.tenant_id
            WHERE c.review_id = 1 AND c.clause_text IS NOT NULL
        """)).fetchall()

        all_tenants = conn.execute(text("""
            SELECT id, tenant_name FROM lease_tenants WHERE review_id = 1
        """)).fetchall()

    tenant_map = {t[1].lower().strip(): t[0] for t in all_tenants}
    tenant_clean = {}
    for t in all_tenants:
        clean = re.sub(r'[#\d]+$', '', t[1]).strip().lower()
        tenant_clean[clean] = (t[0], t[1])

    KNOWN_REFS = {
        "jc penny": None, "jcpenney": None, "jc penney": None,
        "lifetime fitness": None, "ac moore": None,
        "kohl's": "kohl's", "kohls": "kohl's",
        "ross": "ross", "ross dress for less": "ross",
        "petsmart": "petsmart",
        "at home": "at home",
        "sam's club": "sam's club", "sams club": "sam's club",
        "dsw": "dsw",
        "office depot": "office depot",
        "o'reilly": "o'reilly",
        "2nd & charles": "2nd & charles", "2nd and charles": "2nd & charles",
        "bealls": "bealls",
    }

    def find_tenant_id(name):
        nl = name.lower().strip()
        for tn, tid in tenant_map.items():
            if nl in tn or tn.startswith(nl):
                return tid
        for clean, (tid, full) in tenant_clean.items():
            if nl in clean or clean.startswith(nl):
                return tid
        return None

    results = []
    for clause in clauses:
        cot_id, tenant_id, tenant_name, clause_text = clause
        refs_found = set()
        for ref_pattern, canonical in KNOWN_REFS.items():
            if re.search(re.escape(ref_pattern), clause_text, re.IGNORECASE):
                if canonical:
                    refs_found.add(canonical)

        print(f"\n{tenant_name}: {len(refs_found)} co-tenant refs")
        for ref in sorted(refs_found):
            ref_tid = find_tenant_id(ref)
            print(f"  -> {ref} (tenant_id={ref_tid})")
            results.append((cot_id, tenant_id, ref, ref_tid))

    with engine.connect() as conn:
        for cot_id, tenant_id, ref_name, ref_tid in results:
            conn.execute(text("""
                INSERT INTO lease_cotenancy_refs
                    (cotenancy_id, tenant_id, referenced_tenant_name, referenced_tenant_id)
                VALUES (:cid, :tid, :rtn, :rtid)
            """), {'cid': cot_id, 'tid': tenant_id, 'rtn': ref_name, 'rtid': ref_tid})
        conn.commit()
        r = conn.execute(text("SELECT COUNT(*) FROM lease_cotenancy_refs")).fetchone()
        print(f"\nTotal cotenancy refs inserted: {r[0]}")

    # Update cotenancy records with parsed trigger/cure/termination data
    CLAUSE_DETAILS = {
        "2nd & charles": {
            'trigger': '< 4 of 6 anchors open + < 60% GLA of South Shopping Center',
            'cure_days': 365,
            'alt_rent': '2% of Gross Sales',
            'term_right': True,
            'term_days': 30,
            'sunset': 'After 12mo alt rent: terminate (30d notice) OR resume full rent and permanently waive',
            'curable': True,
            'waiver': 'Resume full rent = permanent waiver for that failure',
        },
        "petsmart": {
            'trigger': 'Both initial co-tenants close OR < 60% GLA (initial); < 50% GLA (renewal)',
            'cure_days': 0,
            'alt_rent': '50% of Base Rent',
            'term_right': True,
            'term_days': 90,
            'sunset': 'Must terminate within 30 days after 540-day mark or permanently waive',
            'curable': False,
            'waiver': 'UNCURABLE after 540+30 days -- right permanently waived if not exercised',
        },
        "dollar tree": {
            'trigger': '< 3 of 5 named tenants open',
            'cure_days': 0,
            'alt_rent': '50% of Base Rent',
            'term_right': True,
            'term_days': 30,
            'sunset': '12mo alt rent cap then resume; termination right ongoing until restored',
            'curable': True,
            'waiver': 'Alt rent capped at 12 months, but termination right persists',
        },
        "bealls": {
            'trigger': '4+ of 6 inducement tenants close for 360 days',
            'cure_days': 360,
            'alt_rent': '50% of Fixed Rent',
            'term_right': True,
            'term_days': 0,
            'sunset': 'After 12mo reduced rent: resume full rent OR terminate',
            'curable': True,
            'waiver': 'Binary choice after 12 months',
        },
        "ross": {
            'trigger': "Kohl's (or replacement) closes OR occupancy % below threshold",
            'cure_days': 180,
            'alt_rent': 'Lesser of: Min Rent, 2% Gross Sales (min 50% Min Rent)',
            'term_right': True,
            'term_days': 0,
            'sunset': 'After 18 months substitute rent: terminate or waive and resume',
            'curable': True,
            'waiver': 'Must elect at 18 months or waive',
        },
        "saloncentric": {
            'trigger': '< 60% leasable area occupied for 6 consecutive months',
            'cure_days': 180,
            'alt_rent': '50% of Base Rent',
            'term_right': True,
            'term_days': 0,
            'sunset': 'After 360 days: terminate or resume full rent',
            'curable': True,
            'waiver': 'Standard terminate-or-resume',
        },
        "dsw": {
            'trigger': '< 4 of 6 Key Stores open + < 70% GLA occupied',
            'cure_days': 0,
            'alt_rent': '4% Gross Sales + RE Tax (capped at Min Rent)',
            'term_right': True,
            'term_days': 0,
            'sunset': 'LL can revoke alt rent after 12mo; must pay full rent or terminate',
            'curable': True,
            'waiver': 'LL-initiated revocation forces election',
        },
        "plato's closet": {
            'trigger': '< 4 of 7 named tenants open for 360 days',
            'cure_days': 360,
            'alt_rent': '50% of Base Rent',
            'term_right': True,
            'term_days': 30,
            'sunset': 'Must terminate within 30 days after 360-day period or permanently waive',
            'curable': False,
            'waiver': 'NARROW WINDOW -- 30 days to act or permanent waiver',
        },
        "shoe carnival": {
            'trigger': "Kohl's OR Ross ceases operations",
            'cure_days': 0,
            'alt_rent': '3% of Gross Receipts',
            'term_right': True,
            'term_days': 10,
            'sunset': 'ONE-TIME right after 12 months, 10-day exercise window only',
            'curable': False,
            'waiver': 'MOST RESTRICTIVE -- miss 10-day window = resume full rent permanently',
        },
    }

    with engine.connect() as conn:
        for tn_pattern, d in CLAUSE_DETAILS.items():
            conn.execute(text("""
                UPDATE lease_cotenancy
                SET trigger_description = :td,
                    cure_period_days = :cpd,
                    alt_rent_formula = :arf,
                    termination_right = :tr,
                    termination_notice_days = :tnd,
                    sunset_provision = :sp,
                    is_curable = :ic,
                    waiver_mechanism = :wm
                WHERE tenant_id IN (
                    SELECT id FROM lease_tenants
                    WHERE review_id = 1 AND LOWER(tenant_name) LIKE :tn
                )
            """), {
                'td': d['trigger'], 'cpd': d['cure_days'],
                'arf': d['alt_rent'], 'tr': d['term_right'],
                'tnd': d['term_days'], 'sp': d['sunset'],
                'ic': d['curable'], 'wm': d['waiver'],
                'tn': f'%{tn_pattern}%',
            })
        conn.commit()
        print("\nCotenancy details updated")
