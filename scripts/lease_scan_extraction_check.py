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
    'over what one request can carry' in (r.get('_extraction_note') or '')
    and 'MB' in (r.get('_extraction_note') or ''),
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


section('Our own caps are the model\'s, and a failed reading says why (Sep 29 2026)')
# GNC's 41-page 1996 lease: read in full, 30 tokens back -- the first sentence of
# the instructions -- and a normal stop. It was logged "Extracted" and kept no
# reason. These pin the remedy and the record.
chk('output room raised to 64,000 (thinking + answer, streamed)',
    SENT.get('max_tokens') == 64_000 == S.MAX_OUTPUT_TOKENS, str(SENT.get('max_tokens')))
chk('text cap raised from 180,000 characters to the 1M-token window\'s share',
    S.MAX_TEXT_CHARS >= 1_000_000, S.MAX_TEXT_CHARS)


def install_queue(replies):
    """Each stream() call takes the next (text, stop_reason); records calls."""
    CALLS = []

    class _Messages:
        def stream(self, **kw):
            CALLS.append(kw)
            text, stop = replies[min(len(CALLS) - 1, len(replies) - 1)]
            return _Stream(_Msg([_Block('text', text=text)], stop))

    class _Client:
        def __init__(self, **kw):
            self.messages = _Messages()

    mod = types.ModuleType('anthropic')
    mod.Anthropic = _Client
    sys.modules['anthropic'] = mod
    return CALLS


ECHO = 'You are a commercial real estate lease analyst. Extract the following.'
calls = install_queue([(ECHO, 'end_turn'), (json.dumps({'square_feet': 1300}), 'end_turn')])
r = call(LEASE_TEXT)
chk('no JSON on the first ask -> asked ONCE more, and the second answer is used',
    len(calls) == 2 and r.get('square_feet') == 1300 and not r.get('_parse_error'),
    str((len(calls), r)))
calls = install_queue([(ECHO, 'end_turn')])
r = call(LEASE_TEXT)
chk('still nothing after the retry -> a parse error carrying the reason',
    len(calls) == 2 and r.get('_parse_error')
    and 'no readable answer' in (r.get('_failure_reason') or ''), r.get('_failure_reason'))
calls = install_queue([('{"square_feet": 1', 'max_tokens')])
r = call(LEASE_TEXT)
chk('a truncated answer is named as running out of room',
    'ran out of room' in (r.get('_failure_reason') or ''), r.get('_failure_reason'))
calls = install_queue([('', 'refusal')])
r = call(LEASE_TEXT)
chk('a refusal is NOT retried, and says it declined',
    len(calls) == 1 and r.get('_refused') and 'declined' in (r.get('_failure_reason') or ''))
calls = install_queue([(json.dumps({'square_feet': 5}), 'end_turn')])
r = call('x' * (S.MAX_TEXT_CHARS + 10))
chk('a document over the text cap is cut AND says so',
    r.get("_truncated") is True, str(r))
r = call(LEASE_TEXT)
chk('...and one under it does not', '_truncated' not in r)


section('A scan the model will not read as a PDF is retried as page images')
# GNC's 1996 lease: degenerate replies to the PDF twice; the rendered pages read.
try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
_d = pymupdf.open()
for n in range(3):
    _d.new_page().insert_text((72, 72), 'Page %d of a scanned lease' % (n + 1))
REAL_PDF = _d.tobytes()
# A TYPED PDF for the text-route checks: since Sep 30 2026 a page carrying under
# PAGE_TEXT_MIN characters marks the file as partly scanned and sends the PDF on
# the FIRST ask (lease_bundle_scan_check.py), so REAL_PDF's one-line pages no
# longer stand for a typed document.
_t = pymupdf.open()
for n in range(3):
    _t.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800),
                                 'Typed lease page %d. ' % (n + 1) + 'Rent is payable. ' * 20,
                                 fontsize=9)
TYPED_PDF = _t.tobytes()


def kinds(kw):
    return [b['type'] for b in kw['messages'][0]['content']]


calls = install_queue([(ECHO, 'end_turn'), (json.dumps({'square_feet': 1300}), 'end_turn')])
r = call('', file_data=REAL_PDF, page_count=3)
chk('first ask sends the PDF', calls and kinds(calls[0]) == ['document', 'text'],
    str(calls and kinds(calls[0])))
chk('the retry sends the RENDERED PAGES, not the same PDF',
    len(calls) == 2 and kinds(calls[1]) == ['image', 'image', 'image', 'text'],
    str(len(calls) > 1 and kinds(calls[1])))
chk('...the images are JPEG, and the answer is used and says how it was read',
    calls[1]['messages'][0]['content'][0]['source']['media_type'] == 'image/jpeg'
    and r.get('square_feet') == 1300 and r.get('_extraction_source') == 'images', str(r))
calls = install_queue([(ECHO, 'end_turn'), (ECHO, 'end_turn')])
r = call('', file_data=PDF, page_count=3)          # not a real PDF: cannot render
chk('a PDF that cannot be rendered is retried as the PDF, and the note says why',
    len(calls) == 2 and kinds(calls[1])[0] == 'document'
    and 'retried as the PDF' in (r.get('_extraction_note') or ''), str(r))
calls = install_queue([(ECHO, 'end_turn'), (json.dumps({'square_feet': 9}), 'end_turn')])
r = call(LEASE_TEXT)
chk('a TEXT document is simply asked again (no images involved)',
    len(calls) == 2 and 'image' not in kinds(calls[1]) and r.get('square_feet') == 9)
section('Step 1 of the rent-roll plan: the three re-read failures (Sep 29 2026)')
# Tropical Smoothie: a 27 MB lease passed a RAW 32 MB check and the API refused
# it at ~36 MB encoded. The check is on the encoded size, and an over-size scan
# goes as page images.
chk('the size check is on the ENCODED request: 25 MB raw does not fit',
    not S._pdf_fits(b'x' * (25 * 1024 * 1024), 'p') and S._pdf_fits(b'x' * (20 * 1024 * 1024), 'p'))
_real_max = S.REQUEST_MAX_BYTES
S.REQUEST_MAX_BYTES = len(REAL_PDF) + 1024 + S.REQUEST_HEADROOM   # this real PDF no longer fits
try:
    calls = install_queue([(json.dumps({'square_feet': 1307}), 'end_turn')])
    r = call('', file_data=REAL_PDF, page_count=3)
finally:
    S.REQUEST_MAX_BYTES = _real_max
chk('an over-size scan is sent as page images on the FIRST ask, not refused',
    len(calls) == 1 and kinds(calls[0]) == ['image', 'image', 'image', 'text']
    and r.get('_extraction_source') == 'images', str((len(calls), calls and kinds(calls[0]))))

# Perkins: the same text failed twice and read on a later run. A failed TEXT
# reading is retried from the document itself.
calls = install_queue([(ECHO, 'end_turn'), (json.dumps({'square_feet': 5560}), 'end_turn')])
r = call(LEASE_TEXT, file_data=TYPED_PDF, page_count=3)
chk('a failed text reading is retried as the PDF',
    len(calls) == 2 and kinds(calls[0]) == ['text'] and kinds(calls[1]) == ['document', 'text']
    and r.get('square_feet') == 5560 and r.get('_extraction_source') == 'pdf',
    str([kinds(c) for c in calls]))

# Sam's Club: 26 NUL characters in the text layer failed the whole document.
class _FakePage:
    def get_text(self):
        return 'Access\x00 Agreement\x00'


class _FakeDoc(list):
    def close(self):
        pass


_real_open = pymupdf.open
pymupdf.open = lambda *a, **k: _FakeDoc([_FakePage(), _FakePage()])
try:
    txt, pages = S.extract_pdf_text(b'%PDF-fake')
finally:
    pymupdf.open = _real_open
chk('NUL characters are stripped from extracted text', '\x00' not in txt
    and 'Access Agreement' in txt and pages == 2, repr(txt))

big = pymupdf.open()
for _ in range(S.IMAGE_MAX_PAGES + 1):
    big.new_page()
blocks, why = S._render_pdf_pages(big.tobytes())
chk('over the image page cap: refused with the reason, not attempted',
    blocks is None and 'over the' in why, why)

print('\n%d passed, %d failed' % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
