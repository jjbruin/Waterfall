"""Guardrail: filing a statement opens the chain, and never re-bases one.

Jim, Sep 19 2026: "Shouldn't the seeding process be integrated into loading the
statements function? If the account does not need a seed because reconciled
balances are carried forward, the process should simply save the statement file
in its place and make it readily available when called by the accountant."

Both halves are asserted here, and BOTH DIRECTIONS OF EACH, because either one
passing alone would be worse than useless:

  * an account with nothing reconciled is OPENED by the statement it is given,
  * an account already carrying its balances forward is NOT touched -- and its
    statement is still filed, still listed and still openable.

A rule tested only in the seeding direction is satisfied by seeding everything,
which would silently re-base a reconciled account: the one outcome
`seed_from_statement` exists to prevent.

Run:  .venv\\Scripts\\python.exe scripts\\treasury_seed_on_import_check.py
"""
import os
import sys
import tempfile
import logging

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import treasury_service as ts  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


DB = os.path.join(tempfile.gettempdir(), 'tr_seed.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine('sqlite:///%s' % DB)
ts.ensure_tables(eng)

JUNE = {"period_start": "2026-06-01", "period_end": "2026-06-30",
        "beginning_balance": 500000.00, "ending_balance": 119701.35,
        "credits_total": 0.0, "debits_total": 380298.65,
        "source_file": "30 June 2026 Test One 1111 PNC.pdf",
        "internally_consistent": True}

# ── the period arithmetic the integration rests on ────────────────────────
chk('next_period rolls a month', ts.next_period('202606') == '202607')
chk('...and a year end', ts.next_period('202612') == '202701')
chk('...and refuses a non-period', ts.next_period('26-06') is None)
chk('period_of reads a statement date',
    ts.period_of('2026-06-30') == '202606')


# ── 1. nothing reconciled: filing OPENS the chain ─────────────────────────
ts.create_account('1000001111', entityid='T1', user='t', engine=eng)
res = ts.import_statement(JUNE, '1000001111', engine=eng,
                          file_data=b'%PDF-june-one', user='t')
chk('the statement files', res.get('ok') and not res.get('error'), str(res)[:80])
chk('...and filing OPENED the following month',
    res.get('seeded_period') == '202607', str(res.get('seeded_period')))
chk('...at the statement\'s own ending balance',
    abs((res.get('seeded_amount') or 0) - 119701.35) < 0.005,
    str(res.get('seeded_amount')))

ob = ts.opening_balance('1000001111', '202607', engine=eng)
chk('...so July now HAS an opening, with no second step',
    ob.get('opening') is not None
    and abs(ob['opening'] - 119701.35) < 0.005, str(ob)[:90])
chk('...carried, not read off a statement at reconcile time',
    (ob.get('source') or '').startswith('carried from 202606'),
    str(ob.get('source')))

with eng.connect() as c:
    st, note = c.execute(text(
        "SELECT status, note FROM tr_periods WHERE account_number='1000001111'"
        "  AND period='202606'")).fetchone()
chk('the seeded month is marked seeded, never closed', st == 'seeded', str(st))
chk('...and the note names the file it came from',
    'Test One 1111' in (note or ''), (note or '')[:70])


# ── re-running a folder is ordinary: no duplicate row, no second seed ─────
again = ts.import_statement(JUNE, '1000001111', engine=eng,
                            file_data=b'%PDF-june-one', user='t')
chk('re-importing the same file is not an error', again.get('ok'), str(again)[:70])
chk('...and says it was already filed', again.get('already_filed') is True)
with eng.connect() as c:
    n = c.execute(text("SELECT COUNT(*) FROM tr_statements "
                       " WHERE account_number='1000001111'")).scalar()
chk('...and did not store the statement twice', n == 1, str(n))


# ── 2. already reconciled: filing must NOT re-base it ─────────────────────
ts.create_account('1000002222', entityid='T2', user='t', engine=eng)
ts.seed_opening('1000002222', '202606', 10000.00, 't', eng)
ts.close_period('1000002222', '202606', user='t', engine=eng)
with eng.connect() as c:
    closed_before = c.execute(text(
        "SELECT computed_ending FROM tr_periods WHERE account_number="
        "'1000002222' AND period='202606' AND status='closed'")).scalar()
chk('the second account has a genuinely closed period',
    closed_before is not None, str(closed_before))

JUNE2 = dict(JUNE, ending_balance=88888.88,
             source_file="30 June 2026 Test Two 2222 PNC.pdf")
res2 = ts.import_statement(JUNE2, '1000002222', engine=eng,
                           file_data=b'%PDF-june-two', user='t')
chk('its statement STILL files', res2.get('ok') and not res2.get('error'),
    str(res2)[:80])
chk('...but the chain was NOT re-based from it',
    res2.get('seeded_period') is None, str(res2.get('seeded_period')))
chk('...and the skip says why, rather than passing silently',
    'reconciled period' in (res2.get('seed_skipped') or ''),
    (res2.get('seed_skipped') or '')[:70])
with eng.connect() as c:
    after = c.execute(text(
        "SELECT computed_ending, status FROM tr_periods WHERE account_number="
        "'1000002222' AND period='202606'")).fetchone()
chk('...the closed period is untouched, to the cent',
    abs(float(after[0]) - float(closed_before)) < 0.005 and after[1] == 'closed',
    str(after))


# ── 3. a statement that needs no seed is still KEPT and REACHABLE ─────────
lst = ts.statements(engine=eng)
chk('both statements are listed', len(lst) == 2, str(len(lst)))
chk('...the one that seeded nothing is among them',
    any(r['account_number'] == '1000002222' for r in lst))
chk('...each carries the period it belongs to',
    all(r['period'] == '202606' for r in lst),
    str([r['period'] for r in lst]))
chk('...and says whether its PDF can be opened',
    all(r['has_file'] for r in lst))
chk('...with the account name, so it can be found by who it belongs to',
    all(r['entityid'] in ('T1', 'T2') for r in lst))

one = ts.statements(account_number='1000002222', engine=eng)
chk('a single account can be asked for', len(one) == 1, str(len(one)))
got = ts.statement_file(one[0]['id'], engine=eng)
chk('...and its PDF comes back, for the account that was NOT seeded',
    got.get('data') == b'%PDF-june-two', str(got.get('error'))[:60])

chk('a period filter finds them', len(ts.statements(period='202606',
                                                    engine=eng)) == 2)
chk('...and a period with none says so, rather than returning all',
    ts.statements(period='202512', engine=eng) == [])


# ── 4. the manual seed route still refuses to re-base ─────────────────────
# The button stays for statements filed before this, so its rule still matters.
man = ts.seed_from_statement('1000002222', '202607', 't', engine=eng)
chk('the manual seed still refuses a reconciled account',
    'reconciled period' in (man.get('error') or ''),
    (man.get('error') or '')[:60])

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
