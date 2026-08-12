"""Extract lease terms from PDFs via Claude API for Windsor Square.

Starts with tenants that have co-tenancy clauses (highest priority),
then processes remaining material leases.

The actual lease PDFs are ground truth. The seller-provided documents
(rent roll, Argus model, cotenancy schedule) are validated AGAINST
the lease-extracted terms.
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['FLASK_ENV'] = 'development'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from flask_app import create_app
app = create_app()

with app.app_context():
    from flask_app.db import get_engine
    from flask_app.services.lease_review_service import (
        extract_pdf_text,
        extract_lease_terms_via_api,
    )
    from sqlalchemy import text
    import logging
    logging.basicConfig(level=logging.INFO)

    engine = get_engine()
    review_id = 1
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    # Get documents prioritized: cotenancy tenants first, then material, then rest
    # Only process Original Lease and Amendment documents
    with engine.connect() as conn:
        docs = conn.execute(text("""
            SELECT d.id, d.tenant_id, d.filename, d.file_path,
                   d.doc_type, d.doc_date,
                   t.tenant_name, t.suite, t.has_cotenancy, t.is_material
            FROM lease_documents d
            JOIN lease_tenants t ON t.id = d.tenant_id
            WHERE d.review_id = :rid
            AND d.extraction_status = 'pending'
            AND d.doc_type IN ('Original Lease', 'Amendment')
            ORDER BY
                t.has_cotenancy DESC,
                t.is_material DESC,
                t.annual_rent DESC,
                d.doc_date ASC
        """), {'rid': review_id}).fetchall()

    print(f"Found {len(docs)} lease/amendment documents to extract")
    print(f"  Co-tenancy tenants first, then material leases, then rest\n")

    # Process mode: --all for everything, default = cotenancy tenants only
    mode = 'cotenancy'
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        mode = 'all'
    elif len(sys.argv) > 1 and sys.argv[1] == '--material':
        mode = 'material'

    extracted = 0
    errors = 0
    skipped = 0

    for doc in docs:
        doc_id, tenant_id = doc[0], doc[1]
        filename, file_path = doc[2], doc[3]
        doc_type = doc[4]
        tenant_name, suite = doc[6], doc[7]
        has_cotenancy, is_material = doc[8], doc[9]

        # Filter by mode
        if mode == 'cotenancy' and not has_cotenancy:
            skipped += 1
            continue
        if mode == 'material' and not (has_cotenancy or is_material):
            skipped += 1
            continue

        print(f"  [{extracted+1}] {tenant_name} / {filename}")
        print(f"       Type: {doc_type} | Suite: {suite} | "
              f"Cotenancy: {'Yes' if has_cotenancy else 'No'}")

        try:
            # Step 1: Extract PDF text
            pdf_text, page_count = extract_pdf_text(file_path)
            print(f"       PDF: {page_count} pages, {len(pdf_text):,} chars")

            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE lease_documents
                    SET extracted_text = :txt, page_count = :pc,
                        extraction_status = 'text_extracted'
                    WHERE id = :did
                """), {'txt': pdf_text, 'pc': page_count, 'did': doc_id})
                conn.commit()

            # Step 2: Call Claude API
            terms = extract_lease_terms_via_api(
                pdf_text, tenant_name, suite, doc_type, api_key
            )

            if terms.get('_parse_error'):
                print(f"       WARNING: Could not parse extraction response")
                errors += 1
                continue

            # Step 3: Store results
            with engine.connect() as conn:
                # Update document status
                conn.execute(text("""
                    UPDATE lease_documents
                    SET extraction_status = 'extracted'
                    WHERE id = :did
                """), {'did': doc_id})

                # Update tenant extraction
                conn.execute(text("""
                    UPDATE lease_tenants
                    SET extraction_json = :ej,
                        extraction_status = 'extracted',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :tid
                """), {
                    'tid': tenant_id,
                    'ej': json.dumps(terms, default=str),
                })

                # Store rent steps
                rent_steps = terms.get('rent_steps') or []
                for step in rent_steps:
                    conn.execute(text("""
                        INSERT INTO lease_rent_steps
                            (tenant_id, effective_date, monthly_rent,
                             annual_rent, rent_per_sf, source_doc)
                        VALUES (:tid, :ed, :mr, :ar, :rpsf, :sd)
                    """), {
                        'tid': tenant_id,
                        'ed': step.get('effective_date'),
                        'mr': step.get('monthly_rent'),
                        'ar': step.get('annual_rent'),
                        'rpsf': step.get('rent_per_sf'),
                        'sd': filename,
                    })

                # Store renewal options
                for opt in (terms.get('renewal_options') or []):
                    conn.execute(text("""
                        INSERT INTO lease_options
                            (tenant_id, option_type, option_number,
                             total_options, term_years, notice_days,
                             notice_deadline, rent_terms, auto_renewal,
                             source_doc)
                        VALUES (:tid, 'renewal', :on, :to, :ty,
                                :nd, :ndl, :rt, :ar, :sd)
                    """), {
                        'tid': tenant_id,
                        'on': opt.get('option_number'),
                        'to': opt.get('total_options'),
                        'ty': opt.get('term_years'),
                        'nd': opt.get('notice_days'),
                        'ndl': opt.get('notice_deadline'),
                        'rt': opt.get('rent_terms'),
                        'ar': opt.get('auto_renewal', False),
                        'sd': filename,
                    })

                conn.commit()

            # Print key findings
            sf = terms.get('square_feet')
            exp = terms.get('lease_expiration')
            cot = terms.get('cotenancy', {})
            print(f"       Extracted: SF={sf}, Exp={exp}, "
                  f"Steps={len(rent_steps)}, "
                  f"Options={len(terms.get('renewal_options') or [])}")
            if cot and cot.get('has_clause'):
                refs = cot.get('named_cotenants') or []
                print(f"       Co-tenancy: {len(refs)} named co-tenants: {refs}")
                print(f"         Trigger: {cot.get('trigger_threshold', 'N/A')}")
                print(f"         Alt rent: {cot.get('alt_rent_formula', 'N/A')}")
                print(f"         Curable: {cot.get('is_curable', 'N/A')}")

            extracted += 1

            # Rate limit to avoid API throttling
            time.sleep(0.5)

        except Exception as e:
            print(f"       ERROR: {e}")
            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE lease_documents
                    SET extraction_status = 'error'
                    WHERE id = :did
                """), {'did': doc_id})
                conn.commit()
            errors += 1

    print(f"\n{'='*60}")
    print(f"Extraction complete: {extracted} extracted, {errors} errors, {skipped} skipped")

    # Show stats
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COUNT(*) FROM lease_rent_steps WHERE tenant_id IN "
            "(SELECT id FROM lease_tenants WHERE review_id = :rid)"
        ), {'rid': review_id}).fetchone()
        print(f"  Rent steps in DB: {r[0]}")

        r = conn.execute(text(
            "SELECT COUNT(*) FROM lease_options WHERE tenant_id IN "
            "(SELECT id FROM lease_tenants WHERE review_id = :rid)"
        ), {'rid': review_id}).fetchone()
        print(f"  Options in DB: {r[0]}")

        r = conn.execute(text("""
            SELECT COUNT(*) FROM lease_documents
            WHERE review_id = :rid AND extraction_status = 'extracted'
        """), {'rid': review_id}).fetchone()
        print(f"  Documents extracted: {r[0]}")
