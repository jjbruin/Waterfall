"""Guardrail: part-scanned files, bundled instruments, and annual option increases.

New business, Sep 30 2026, on the Market at Poplar acceptance: Hobby Lobby's 2024
4th Amendment, Muddy Paws' 1st and 2nd Amendments ("included at the bottom of the
Lease PDF") and Perkins' option rents ("page 2 of the 1st Amendment") had all been
uploaded and were not captured -- "extraction issues rather than missing
documents". Measured on production the same day, all three are PARTLY SCANNED:

  HOBBY LOBBY 4th Amend  4 pages, text per page [54, 54, 54, 289] -- a dotloop
                         signature stamp and nothing else
  PERKINS 1st Amend      [2722, 0, 0, 0, 0] -- page 1 typed, the rest scanned
  MUDDY PAWS lease       22 of 26 pages empty; the typed pages 23-26 ARE the two
                         amendments

Each passed the 200-character whole-document test on its typed pages, so it was
read as text and the scanned pages never reached the model. And Muddy Paws' file,
once read whole, is a lease WITH its amendments: filed as an Original Lease dated
2020, its 2025 extension was laid down first, overwritten by the 2021
commencement letter, and its exercise discarded as "a lease cannot record its own
exercise".

Peak Potential: "two -- five (5) year options that will each have two percent (2%)
annual increases" printed one figure per option; new business wants Year 1
61,287, Year 2 62,513 ... Year 5 66,339.

No API calls. Run:  .venv\\Scripts\\python.exe scripts\\lease_bundle_scan_check.py
"""
import json
import logging
import os
import sys
import tempfile
import types

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

import pymupdf  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import lease_review_service as S  # noqa: E402
from flask_app.services.lease_terms import bundled_instruments  # noqa: E402
from flask_app.services.lease_timeline import build_timeline  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % (detail,) if detail and not cond else ''))


def section(t):
    print('\n--- ' + t)


# ------------------------------------------------------------ routing (stubbed)
SENT = {}


class _B:
    def __init__(self, t, **kw):
        self.type = t
        self.__dict__.update(kw)


class _Stream:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        m = types.SimpleNamespace()
        m.stop_reason = 'end_turn'
        m.content = [_B('thinking', thinking='.'), _B('text', text='{"square_feet": 1}')]
        return m


class _Client:
    def __init__(self, **kw):
        self.messages = self

    def stream(self, **kw):
        SENT.clear()
        SENT.update(kw)
        return _Stream()


sys.modules['anthropic'] = types.SimpleNamespace(Anthropic=_Client)

TYPED = 'The Landlord and the Tenant agree that the Minimum Annual Rent shall be ' * 4


def pdf(pages):
    d = pymupdf.open()
    for body in pages:
        p = d.new_page()
        if body:
            p.insert_textbox(pymupdf.Rect(40, 40, 560, 800), body, fontsize=9)
    return d.tobytes()


MIXED = pdf([TYPED, '', '', ''])          # Perkins' shape: page 1 typed, 2-4 scanned
ALL_TYPED = pdf([TYPED, TYPED])
LONG_TEXT = TYPED * 3


def route(file_data):
    r = S.extract_lease_terms_via_api(LONG_TEXT, 'T', 'S1', 'Amendment', api_key='k',
                                      file_data=file_data, page_count=4)
    kinds = [b['type'] for b in SENT['messages'][0]['content']]
    return r, kinds


section('A partly scanned file is read from the PDF')
chk('pages_without_text counts the scanned pages', S.pages_without_text(MIXED) == (3, 4),
    S.pages_without_text(MIXED))
r, kinds = route(MIXED)
chk('the PDF goes with the prompt, though the text is long', kinds == ['document', 'text'],
    kinds)
chk('...the route says pdf, and the note says why',
    r.get('_extraction_source') == 'pdf'
    and '3 of 4 pages carry no text layer' in (r.get('_extraction_note') or ''), r)
section('...and nothing else changes route')
r, kinds = route(ALL_TYPED)
chk('a fully typed file still goes as text', kinds == ['text']
    and r.get('_extraction_source') == 'text', kinds)
r, kinds = route(b'%PDF-1.4 not really a pdf')
chk('a file that cannot be opened is not taken for a scan', kinds == ['text'], kinds)
chk('pages_without_text of no file is (0, 0)', S.pages_without_text(None) == (0, 0))
prompt = S.EXTRACTION_PROMPT
chk('the prompt asks for the instruments in the file', '"instruments"' in prompt
    and 'AS THEY STAND AFTER THE LAST INSTRUMENT' in prompt)
chk('the prompt asks how often an option increase applies',
    'escalation_frequency' in prompt)
src = open(os.path.join('flask_app', 'services', 'lease_review_service.py'),
           encoding='utf-8').read()
chk('the extraction loop always loads the PDF for a term-bearing document',
    'if len((pdf_text or \'\').strip()) < SCAN_TEXT_THRESHOLD:\n                        got' not in src)

# ------------------------------------------------------------ bundles
section('bundled_instruments')
BUNDLE = {'instruments': [
    {'type': 'Original Lease', 'title': 'LEASE AGREEMENT', 'date': '2020-06-10'},
    {'type': 'Amendment', 'title': 'FIRST LEASE AMENDMENT', 'date': '2021-07-01'},
    {'type': 'Amendment', 'title': 'SECOND LEASE AMENDMENT', 'date': '2025-02-01'}]}
b = bundled_instruments(BUNDLE)
chk('a lease with two amendments bound in is a bundle, dated by its last',
    b and b['last_date'] == '2025-02-01' and len(b['later']) == 2, b)
chk('one instrument is not a bundle', bundled_instruments(
    {'instruments': BUNDLE['instruments'][:1]}) is None)
chk('a lease with only exhibits after it is not a bundle', bundled_instruments(
    {'instruments': [BUNDLE['instruments'][0],
                     {'type': 'Other', 'title': 'EXHIBIT A SITE PLAN'}]}) is None)
chk('no list is not a bundle', bundled_instruments({'lease_expiration': '2030-01-01'}) is None)
b = bundled_instruments({'instruments': [BUNDLE['instruments'][0],
                                         {'type': 'Amendment', 'title': 'FIRST AMENDMENT'}]})
chk('undated amendments are still a bundle, with no date to layer by',
    b and b['last_date'] == '2020-06-10', b)

DB = os.path.join(tempfile.gettempdir(), 'lease_bundle_scan.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
S.ensure_lease_tables(eng)

EXERCISED = [{'option_number': 1, 'term_years': 3, 'option_start': '2025-06-01',
              'option_end': '2028-05-31', 'exercised': True}]
LETTER = ('2021.04.16-Muddy Paws-Commencement Letter.pdf', 'Commencement Letter',
          '2021-04-16', {'lease_commencement': '2021-04-16',
                         'rent_commencement': '2021-05-16',
                         'lease_expiration': '2025-05-31'})
AS_AMENDED = {'lease_expiration': '2028-05-31', 'lease_commencement': '2020-07-01',
              'rent_commencement': '2020-08-01', 'renewal_options': EXERCISED}
TENANTS = {
    1: ('Muddy Paws (bundle)', [
        ('2020.06.10-Muddy Paws-Lease Agmt-Poplar.pdf', 'Original Lease', '2020-06-10',
         dict(AS_AMENDED, **BUNDLE)), LETTER]),
    # THE SAME READING WITHOUT THE INSTRUMENT LIST: a genuine original lease that
    # claims its own exercise. The old rules must still hold for it.
    2: ('Muddy Paws (lease alone)', [
        ('2020.06.10-Muddy Paws-Lease Agmt-Poplar.pdf', 'Original Lease', '2020-06-10',
         dict(AS_AMENDED)), LETTER]),
}
with eng.begin() as c:
    c.execute(text("INSERT INTO lease_reviews (id, property_name) VALUES (1, 'Poplar')"))
    did = 0
    for tid, (name, docs) in TENANTS.items():
        c.execute(text("INSERT INTO lease_tenants (id, review_id, tenant_name, tenant_status)"
                       " VALUES (:i, 1, :n, 'active')"), {'i': tid, 'n': name})
        for fn, dt, dd, terms in docs:
            did += 1
            c.execute(text(
                "INSERT INTO lease_documents (id, tenant_id, review_id, filename, doc_type,"
                " doc_date, extraction_status, extraction_json) VALUES"
                " (:i,:t,1,:f,:dt,:dd,'extracted',:j)"),
                {'i': did, 't': tid, 'f': fn, 'dt': dt, 'dd': dd, 'j': json.dumps(terms)})
C = {tid: S.consolidate_tenant_extractions(eng, tid) for tid in TENANTS}

section('Muddy Paws: a lease with its amendments bound in')
chk('the expiration is the 2nd Amendment\'s 2028-05-31, not the letter\'s 2025',
    C[1]['lease_expiration'] == '2028-05-31', C[1]['lease_expiration'])
chk('...the exercised extension stands, so no option remains',
    C[1].get('_options_summary') == 'None', C[1].get('_options_summary'))
chk('...the file is layered LAST, after the 2021 letter',
    (C[1].get('_governing_document') or '').startswith('2020.06.10'),
    C[1].get('_documents_applied'))
chk('...the commencement dates are still the letter\'s (a bundle only fills them)',
    C[1].get('lease_commencement') == '2021-04-16'
    and C[1].get('rent_commencement') == '2021-05-16',
    (C[1].get('lease_commencement'), C[1].get('rent_commencement')))
chk('...and the reader is told why it was layered there',
    any('SECOND LEASE AMENDMENT' in n and '2025-02-01' in n
        for n in C[1].get('_order_notes') or []), C[1].get('_order_notes'))
with eng.connect() as c:
    stored = c.execute(text("SELECT doc_type, doc_date FROM lease_documents WHERE id = 1")).fetchone()
chk('...the stored document is not re-typed or re-dated',
    tuple(stored) == ('Original Lease', '2020-06-10'), tuple(stored))
section('...and a lease on its own is treated as before')
chk('an original lease cannot record its own exercise',
    C[2].get('_options_summary') == '1 x 3 Years', C[2].get('_options_summary'))
chk('...so the dated letter after it governs the expiration',
    C[2]['lease_expiration'] == '2025-05-31', C[2]['lease_expiration'])

# ------------------------------------------------------------ annual option increases
section('Peak Potential: 2% a year, every year of each option')
WORDING = ('two - five (5) year options that will each have two percent (2%) '
           'annual increases')


def peak(entry_extra, wording=WORDING):
    opts = [{'option_number': n, 'term_years': 5, 'option_start': s, 'option_end': e,
             'rent_basis': 'stated_increase',
             'rent_schedule': [dict({'period': wording, 'start': s, 'end': e,
                                     'annual_rent': None, 'escalation_pct': 2},
                                    **entry_extra)]}
            for n, s, e in ((1, '2029-02-01', '2034-01-31'), (2, '2034-02-01', '2039-01-31'))]
    terms = {'lease_commencement': '2019-02-01', 'lease_expiration': '2029-01-31',
             '_remaining_options': opts}
    steps = [{'effective_date': '2028-02-01', 'annual_rent': 60085.56}]
    return build_timeline(terms, steps, 3600, '2026-09-01')


tl = peak({})
o1 = tl['options'][0]['periods']
chk('option 1 prints five yearly periods', len(o1) == 5, len(o1))
chk('...61,287 / 62,513 / 63,763 / 65,038 / 66,339, as new business gave them',
    [p['annual_rent'] for p in o1] == [61287, 62513, 63763, 65038, 66339],
    [p['annual_rent'] for p in o1])
chk('...each a year long, the last ending on the option\'s end',
    o1[0]['start'] == '2029-02-01' and o1[0]['end'] == '2030-01-31'
    and o1[-1]['start'] == '2033-02-01' and o1[-1]['end'] == '2034-01-31',
    [(p['start'], p['end']) for p in o1])
chk('...and each says it was derived', all(p['derived'] for p in o1))
o2 = tl['options'][1]['periods']
chk('option 2 carries on from option 1\'s last year (66,339 x 1.02)',
    len(o2) == 5 and round(o2[0]['annual_rent']) == 67666, [p['annual_rent'] for p in o2])
chk('no option-period mismatch is raised',
    not [f for f in tl['flags'] if f['code'] == 'option_period_mismatch'], tl['flags'])
opt_rows = [r for r in tl['rows'] if r['kind'] in ('option', 'option_step')]
chk('the exhibit rows carry all ten years', len(opt_rows) == 10, len(opt_rows))
chk('the extraction\'s own frequency is read first: annual with no wording',
    len(peak({'escalation_frequency': 'annual'}, wording='Option period')['options'][0]
        ['periods']) == 5)
once = peak({'escalation_frequency': 'once'})['options'][0]['periods']
chk('...and "once" is one increase for the whole option, whatever the wording says',
    len(once) == 1 and round(once[0]['annual_rent']) == 61287, once)
legacy = peak({}, wording='Option period')['options'][0]['periods']
chk('a percentage with neither field nor wording is applied once, as before',
    len(legacy) == 1, legacy)

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
