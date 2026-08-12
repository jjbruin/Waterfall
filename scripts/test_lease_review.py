"""Full end-to-end test of the lease review system for Windsor Square."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['FLASK_ENV'] = 'development'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app import create_app
app = create_app()

with app.app_context():
    from flask_app.db import get_engine
    from flask_app.services.lease_review_service import (
        get_expiration_histogram,
        get_cotenancy_matrix,
        get_scenario_analysis,
        generate_lease_review_excel,
    )
    from sqlalchemy import text

    engine = get_engine()
    review_id = 1

    # === EXPIRATION HISTOGRAM ===
    hist = get_expiration_histogram(engine, review_id)
    print("=" * 90)
    print("LEASE EXPIRATION SCHEDULE -- Windsor Square, Matthews NC")
    print(f"Total GLA: {hist['totals']['total_gla']:,.0f} SF")
    print(f"Total Annual Rent: ${hist['totals']['total_annual_rent']:,.0f}")
    print("=" * 90)
    print()
    print(f"{'Year':>6}  {'Expiring SF':>12}  {'Expiring Rent':>14}  {'% Total':>8}  {'Avg $/SF':>9}  {'Tenants':>8}")
    print("-" * 70)
    for yr in hist['yearly_data']:
        if yr['tenant_count'] > 0 or yr['year'] <= 2036:
            bar = "#" * int(yr['pct_of_total_rent'] / 1.5)
            print(f"{yr['year']:>6}  {yr['expiring_sf']:>12,.0f}  ${yr['expiring_rent']:>12,.0f}  "
                  f"{yr['pct_of_total_rent']:>7.1f}%  ${yr['avg_rent_per_sf']:>7.2f}  {yr['tenant_count']:>8}  {bar}")

    print()
    print("=" * 90)
    print("MATERIAL LEASES MATURING BY YEAR")
    print("=" * 90)
    for yr in sorted(hist['material_leases'].keys()):
        print(f"\n  {yr}:")
        for m in hist['material_leases'][yr]:
            cot = "  ** CO-TENANCY IMPLICATION **" if m['has_cotenancy'] else ""
            print(f"    {m['tenant_name']:35s} {m['suite']:12s} {m['square_feet']:>8,.0f} SF  "
                  f"${m['annual_rent']:>10,.0f}  ${m['rent_per_sf']:>.2f}/SF{cot}")
            if m.get('cotenancy_implication'):
                print(f"      {m['cotenancy_implication']}")

    # === COTENANCY ANALYSIS ===
    matrix = get_cotenancy_matrix(engine, review_id)
    print()
    print("=" * 90)
    print("CO-TENANCY RISK ANALYSIS -- Departing Tenant Impact")
    print("=" * 90)
    for cotenant, risk in sorted(matrix['rent_at_risk'].items(),
                                  key=lambda x: x[1]['total_dependent_rent'],
                                  reverse=True):
        print(f"\n  If {cotenant.upper()} departs:")
        print(f"    {risk['dependent_count']} tenants affected  |  "
              f"${risk['total_dependent_rent']:,.0f} annual rent at risk  |  "
              f"{risk['termination_eligible_count']} can terminate")
        for dep in risk['dependents']:
            term = " [CAN TERMINATE]" if dep.get('termination_right') else ""
            print(f"      {dep['dependent_tenant']:30s} ${dep.get('annual_rent', 0):>10,.0f}  "
                  f"{dep.get('alt_rent', ''):40s}{term}")

    # === SCENARIO ANALYSIS ===
    scenarios = get_scenario_analysis(engine, review_id)
    print()
    print("=" * 90)
    print("CASCADING SCENARIO ANALYSIS")
    print("=" * 90)
    for s in scenarios[:5]:
        print(f"\n  SCENARIO: {s['departing_tenant'].upper()} DEPARTS")
        print(f"  Total rent at risk: ${s['total_dependent_rent']:,.0f}")
        print(f"  Tenants that can terminate: {s['termination_eligible']}")
        print(f"  {'Tenant':30s} {'Annual Rent':>12s}  {'Alt Rent Formula':40s} {'Cure':15s} {'Termination'}")
        print(f"  {'-' * 110}")
        for imp in s['impacts']:
            cure = 'UNCURABLE' if not imp['is_curable'] else f"{imp.get('cure_days', '?')}d cure"
            term = 'CAN TERMINATE' if imp['can_terminate'] else 'No termination'
            print(f"  {imp['tenant']:30s} ${imp['annual_rent']:>10,.0f}  "
                  f"{(imp.get('alt_rent_formula') or 'N/A'):40s} {cure:15s} {term}")

    # === DATABASE STATS ===
    print()
    print("=" * 90)
    print("DATABASE STATISTICS")
    print("=" * 90)
    with engine.connect() as conn:
        tables = ['lease_reviews', 'lease_tenants', 'lease_documents',
                  'lease_rent_steps', 'lease_cotenancy', 'lease_cotenancy_refs',
                  'lease_exclusive_use', 'lease_options', 'lease_validation']
        for t in tables:
            r = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).fetchone()
            print(f"  {t:30s} {r[0]:>6} rows")

    # === GENERATE EXCEL ===
    excel_bytes = generate_lease_review_excel(engine, review_id)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'Windsor_Square_Lease_Review.xlsx')
    with open(out_path, 'wb') as f:
        f.write(excel_bytes)
    print(f"\n  Excel workbook: {len(excel_bytes):,} bytes -> {out_path}")
    print("  Sheets: Executive Summary, Lease Expirations, Rent Roll Validation,")
    print("          Co-Tenancy Detail, Exclusive Use, Option Schedule")
