"""Guardrail: a statement whose account is unknown is HELD, not lost.

Jim, Sep 19 2026: "for the statements without a production account, I would like
you to create a record and prompt the user to find and input the account number for
future matching of the data pulls." And: accounting should be able to pull up a copy
of the statement from the treasury screen.

An account registers itself from an activity import, and PNC serves only 90 days --
so an account quiet longer than that has a statement showing real money and no
transaction anywhere to introduce it. Of the 64 real June 2026 statements, 14 land
here and two hold money (PPI Life Storage NY 119,701.35, PSC Ambassadors Fund TGA VI
629,125.04). The old behaviour parsed them correctly, said so in a result row, and
then kept nothing.

THE CHECK THAT MATTERS MOST is that a typed number is validated AGAINST THE MASK.
Without it a mistyped digit registers a plausible new account, the statement files
against it, and when the real account arrives under its true number the balance is
split across two records with nothing saying so.

Driven by a REAL statement PDF. Skips with a reason where that file is absent, so it
still runs in the container.

Run:  .venv\Scripts\python.exe scripts\treasury_pending_check.py
"""
import os
import sys
import tempfile
import logging

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

import pdfplumber  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from flask_app.services import treasury_service as ts  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name + (f'  [{detail}]' if detail else ''))


DB = os.path.join(tempfile.gettempdir(), 'tr_e2e.db')
if os.path.exists(DB):
    os.remove(DB)
eng = create_engine(f'sqlite:///{DB}')
ts.ensure_tables(eng)

F = (r"C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents\2026\06.2026"
     r"\30 June 2026 PPI Life Storage NY LLC 7891 PNC.pdf")
if not os.path.exists(F):
    print('  SKIP  the real June statement is not on this machine')
    print('0 passed, 0 failed, 1 skipped')
    sys.exit(0)
raw = open(F, 'rb').read()
with pdfplumber.open(F) as pdf:
    txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
parsed = ts.parse_statement_text(txt, source_file=os.path.basename(F))
chk('the real statement parses', parsed.get('ending_balance') == 119701.35,
    str(parsed.get('ending_balance')))
chk('...and carries its mask', parsed.get('account_suffix') == 'XX-XXXX-7891',
    str(parsed.get('account_suffix')))

# No account registered -> it must be HELD, with the PDF.
chk('no account matches it yet',
    ts.match_account_by_suffix(parsed['account_suffix'], engine=eng).get('error')
    is not None)
held = ts.hold_unmatched_statement(parsed, engine=eng, file_data=raw)
chk('it is held rather than refused', held.get('ok') and held.get('held'),
    str(held))
pend = ts.pending_statements(engine=eng)
chk('it appears in the prompt list', len(pend) == 1, str(len(pend)))
chk('...with the balance somebody needs to see',
    pend[0]['ending_balance'] == 119701.35)
chk('...and the entity name off the filename, as a hint',
    pend[0]['statement_name'] == 'PPI Life Storage NY LLC',
    pend[0]['statement_name'])
chk('...and the mask, which is what lets a typed number be checked',
    pend[0]['account_suffix'] == 'XX-XXXX-7891')

# Re-importing the same file must not stack duplicates of one question.
ts.hold_unmatched_statement(parsed, engine=eng, file_data=raw)
chk('re-importing does not duplicate the question',
    len(ts.pending_statements(engine=eng)) == 1)

# The accountant can open the PDF to FIND the number.
got = ts.statement_file(pend[0]['id'], pending=True, engine=eng)
chk('the PDF can be pulled up while it is still pending',
    got.get('data', b'')[:4] == b'%PDF', str(got.get('error'))[:60])
chk('...and it is the file that was uploaded', got.get('data') == raw)

# A number that does not fit the mask is REFUSED.
bad = ts.resolve_pending_statement(pend[0]['id'], '9999999999', user='t', engine=eng)
chk('a number that does not fit the mask is refused',
    'does not fit' in (bad.get('error') or ''), str(bad.get('error'))[:70])
chk('...and nothing was registered',
    len(ts.accounts(engine=eng) if hasattr(ts, 'accounts') else []) == 0
    or all(a['account_number'] != '9999999999'
           for a in ts.accounts(engine=eng)))
chk('...and it is still pending', len(ts.pending_statements(engine=eng)) == 1)

# A well-formed number resolves: account registered, statement filed.
good = ts.resolve_pending_statement(pend[0]['id'], '8517897891',
                                    entityid='PPILS', user='t', engine=eng)
chk('a number fitting the mask is accepted', good.get('ok'), str(good)[:90])
chk('...the account is registered',
    any(a['account_number'] == '8517897891' for a in ts.accounts(engine=eng)))
chk('...the statement is filed',
    eng.connect().execute(text(
        "SELECT COUNT(*) FROM tr_statements WHERE account_number='8517897891'"
    )).scalar() == 1)
chk('...with its balance intact',
    abs(eng.connect().execute(text(
        "SELECT ending_balance FROM tr_statements WHERE account_number="
        "'8517897891'")).scalar() - 119701.35) < 0.005)
chk('...and the PDF carried across, not left behind',
    ts.statement_file(eng.connect().execute(text(
        "SELECT id FROM tr_statements WHERE account_number='8517897891'"
    )).scalar(), engine=eng).get('data') == raw)
chk('...and it is no longer prompting', len(ts.pending_statements(engine=eng)) == 0)
chk('...but the record is kept, showing who placed it',
    ts.pending_statements(include_resolved=True, engine=eng)[0]['resolved_account']
    == '8517897891')

# THE POINT OF ASKING: the next pull routes by itself.
nxt = ts.match_account_by_suffix('XX-XXXX-7891', engine=eng)
chk('a LATER statement for the same account now routes automatically',
    nxt.get('account_number') == '8517897891', str(nxt))

print(f'\n{len(OK)} passed, {len(BAD)} failed')
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
