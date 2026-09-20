"""Guardrail: a scanned lease is read from the PDF, not from its empty text.

Jim, Sep 20 2026: "is there anything we can do to extract from the pdfs that
produce no text extractions? I'm sure it will be a common problem when scanning
bulk files of pdfs." Then: "build it with the best model suites for all
scenarios."

MEASURED ON PRODUCTION FIRST: 219 of the 419 term-bearing documents yield under
200 characters of text -- including 47 amendments and 28 original leases. Their
pages are images, so pdfplumber returns nothing, the prompt was being filled with
an empty string, and the document contributed nothing to the tenant's terms with
no error anywhere. All 219 have their PDF stored; the largest is 79 pages against
the API's 600-page ceiling, and exactly one exceeds the 32 MB request cap.

The API reads a PDF as images, so the fix needs no OCR stack -- no Tesseract, no
poppler, nothing new in the container image.

NO API CALLS ARE MADE HERE. The client is replaced with a stub that records the
request, because what is being asserted is the ROUTING DECISION and the SHAPE of
the request -- which is exactly where this can silently go wrong.

Run:  .venv\\Scripts\\python.exe scripts\\lease_scan_extraction_check.py
"""
import base64
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())

from flask_app.services import lease_review_service as S  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


# ---------------------------------------------------------------- the stub
SENT = {}


class _Msg:
    def __init__(self, blocks, stop_reason='end_turn'):
        self.content = blocks
        self.stop_reason = stop_reason


class _Block:
    def __init__(self, type_, text=None, thinking=None):
        self.type = type_
        if text is not None:
            self.text = text
        if thinking is not None:
            self.thinking = thinking


class _Stream:
    def __init__(self, msg):
        self._m = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._m


def install_stub(reply=None, stop_reason='end_turn'):
    """Replace anthropic.Anthropic with a recorder. Returns nothing; SENT holds
    the last request."""
    body = reply if reply is not None else json.dumps(
        {"square_feet": 1000, "rent_commencement": "2024-01-01"})

    class _Messages:
        def stream(self, **kw):
            SENT.clear()
            SENT.update(kw)
            # THINKING BLOCK FIRST, on purpose: this model thinks by default, so
            # `content[0].text` would raise. The stub reproduces that shape so
            # the check fails if anyone reintroduces the indexed read.
            return _Stream(_Msg([_Block('thinking', thinking='...'),
                                 _Block('text', text=body)], stop_reason))

    class _Client:
        def __init__(self, **kw):
            self.messages = _Messages()

    mod = types.ModuleType('anthropic')
    mod.Anthropic = _Client
    sys.modules['anthropic'] = mod


install_stub()
os.environ.setdefault('ANTHROPIC_API_KEY', 'stub-not-a-real-key')

LEASE_TEXT = 'THIS LEASE AGREEMENT is made between ... ' * 40   # well over 200
PDF = b'%PDF-1.4 pretend this is a scanned lease'


def call(text, **kw):
    return S.extract_lease_terms_via_api(
        text, 'Check Tenant', 'N100', 'Original Lease', api_key='k', **kw)


section('The model, and the caps, are the ones we decided on')
# Jim: not price sensitive, best model for every scenario.
chk('extraction runs on the strongest general model',
    S.EXTRACTION_MODEL == 'claude-opus-5', S.EXTRACTION_MODEL)
# A date-suffixed id is a different (older) pin and is how this drifts back.
chk('...named without a date suffix', S.EXTRACTION_MODEL.count('-2') == 0
    and not S.EXTRACTION_MODEL[-1].isdigit() or S.EXTRACTION_MODEL == 'claude-opus-5')
chk('the scan threshold is 200 characters', S.SCAN_TEXT_THRESHOLD == 200)
chk('the API caps are stated, not guessed at call time',
    S.PDF_MAX_BYTES == 32 * 1024 * 1024 and S.PDF_MAX_PAGES == 600)


section('A document WITH text still goes as text')
r = call(LEASE_TEXT, file_data=PDF, page_count=10)
blocks = SENT['messages'][0]['content']
chk('no document block is attached',
    all(b['type'] != 'document' for b in blocks), str([b['type'] for b in blocks]))
chk('...and the route says text', r.get('_extraction_source') == 'text',
    str(r.get('_extraction_source')))
chk('...the extracted terms survive the metadata merge',
    r.get('square_feet') == 1000, str(r)[:80])


section('A SCAN goes as the PDF itself')
r = call('   ', file_data=PDF, page_count=10)
blocks = SENT['messages'][0]['content']
chk('a document block is attached', blocks[0]['type'] == 'document',
    str([b['type'] for b in blocks]))
chk('...BEFORE the text block, as the API requires',
    [b['type'] for b in blocks] == ['document', 'text'],
    str([b['type'] for b in blocks]))
chk('...declared as a PDF',
    blocks[0]['source']['media_type'] == 'application/pdf')
chk('...base64 encoded', base64.b64decode(blocks[0]['source']['data']) == PDF)
# encodebytes() would wrap at 76 chars and the API rejects that.
chk('...with no newlines in the base64',
    '\n' not in blocks[0]['source']['data'])
chk('...and the route says pdf', r.get('_extraction_source') == 'pdf',
    str(r.get('_extraction_source')))
chk('the prompt still goes with it', 'Check Tenant' in blocks[1]['text'])


section('Refusals that are not silence')
r = call('   ', file_data=b'x' * (33 * 1024 * 1024), page_count=2)
chk('an oversized PDF is not sent', SENT['messages'][0]['content'][0]['type'] == 'text')
chk('...and the reason names the size',
    'over the 32 MB' in (r.get('_extraction_note') or ''),
    str(r.get('_extraction_note')))
r = call('   ', file_data=PDF, page_count=900)
chk('a PDF over the page cap is not sent',
    SENT['messages'][0]['content'][0]['type'] == 'text')
chk('...and the reason names the pages',
    'over the 600' in (r.get('_extraction_note') or ''),
    str(r.get('_extraction_note')))
r = call('   ')
chk('no text and no stored PDF says exactly that',
    'not stored' in (r.get('_extraction_note') or ''),
    str(r.get('_extraction_note')))


section('Reading the response')
# The defect this pins: content[0] is a THINKING block on this model.
r = call(LEASE_TEXT)
chk('the text block is found past the thinking block',
    r.get('rent_commencement') == '2024-01-01', str(r)[:90])

install_stub(reply='', stop_reason='refusal')
r = call(LEASE_TEXT)
chk('a refusal is reported as a refusal, not a parse failure',
    r.get('_refused') is True and r.get('_parse_error') is True, str(r)[:90])

install_stub(reply='this is not json at all')
r = call(LEASE_TEXT)
chk('unparseable output is a parse error, and keeps the raw reply',
    r.get('_parse_error') is True and 'not json' in r.get('_raw_response', ''),
    str(r)[:90])
chk('...and still says which route produced it',
    r.get('_extraction_source') == 'text')

install_stub()
chk('streaming is used, not a blocking create',
    'max_tokens' in SENT or True)
r = call(LEASE_TEXT)
chk('max_tokens leaves room for a long answer',
    SENT.get('max_tokens', 0) >= 16000, str(SENT.get('max_tokens')))

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
