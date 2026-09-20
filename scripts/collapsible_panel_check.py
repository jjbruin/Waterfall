"""Guardrail: the input column folds away, and folding it actually frees the width.

Jim, Sep 19 2026: "many of our pages have input sections on the left and analysis
sections on the right. Can we provide the same little arrow as exists in the side
bar to expand or collapse the input sections throughout the app? This would make
more room on the screen for the analysis."

THE FAILURE THIS EXISTS TO CATCH IS SILENT. A GRID parent sets the column track,
so a child that narrows itself to 30px reclaims NOTHING -- the track stays at
290px and 260px of empty space sits where the panel was. The arrow works, the
panel disappears, and the analysis is exactly as cramped as before. Nothing
errors and nothing on screen says so. So every grid page is asserted to declare a
COLLAPSED TRACK as well as to use the component.

Measured in the running app at 1440px before this was written, both directions:

  Reports        sidebar 280 -> 30,  results  888 -> 1138
  Ownership      picker  290 -> 30,  chain    789 -> 1064
  Data Explorer  tables  240 -> 30,  grid     833 -> 1043
  Workpapers     steps   320 -> 30,  evidence 721 -> 1011
  Prospect       setup   480 -> 33,  results  657 -> 1119

Run:  .venv\\Scripts\\python.exe scripts\\collapsible_panel_check.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS = os.path.join(ROOT, 'vue_app', 'src', 'views')
COMP = os.path.join(ROOT, 'vue_app', 'src', 'components', 'common',
                    'CollapsiblePanel.vue')

OK, BAD, SKIP = [], [], []


def chk(name, cond, detail=''):
    (OK if cond else BAD).append(name)
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + ('  [%s]' % detail if detail else ''))


def section(t):
    print('\n--- ' + t)


#: view -> (the panel's class, whether its parent sizes the track itself)
PAGES = {
    'OwnershipView.vue':        ('picker', 'grid'),
    'WorkpapersView.vue':       ('steps', 'grid'),
    'ReportsView.vue':          ('reports-sidebar', 'flex'),
    'DataExplorerView.vue':     ('table-list-panel', 'flex'),
    'ProspectAnalysisView.vue': ('setup-panel', 'flex'),
}

if not os.path.isdir(VIEWS):
    print('  SKIP  Vue source is not in this image')
    print('0 passed, 0 failed, 1 skipped')
    sys.exit(0)

section('The component itself')
src = open(COMP, encoding='utf-8').read()
chk('CollapsiblePanel exists', bool(src))
# The sidebar's own control is a bare < / >. Matching it was the request.
chk('...and uses the sidebar\'s own arrow', "'>' : '<'" in src)
# A control that disappears when used cannot be undone by someone who did not
# already know it was there -- v482 shipped exactly that and had to fix it.
chk('the toggle survives collapsing, so it can be undone',
    'cpanel.collapsed .cpanel-toggle' in src)
chk('...and the collapsed rail still names the panel',
    'cpanel-rail' in src and 'v-if="collapsed"' in src)
chk('the rail is clickable too, not just the arrow',
    re.search(r'cpanel-rail"[^>]*@click="toggle"', src) is not None)
# Storage is a per-viewer convenience: it must never be load-bearing.
chk('the remembered choice is wrapped in try/catch both ways',
    src.count('try {') >= 2 and src.count('catch') >= 2)
chk('...and a panel with no storageKey simply opens',
    "storageKey: ''" in src and 'if (!KEY) return' in src)
chk('the collapsed width is stated for flex AND width parents',
    'flex: 0 0 30px' in src and 'max-width: 30px' in src)

section('Every two-column page uses it')
for fname, (cls, kind) in sorted(PAGES.items()):
    p = os.path.join(VIEWS, fname)
    if not os.path.exists(p):
        SKIP.append(fname)
        print('  SKIP  %s is not present' % fname)
        continue
    s = open(p, encoding='utf-8').read()
    chk('%s imports the panel' % fname, 'CollapsiblePanel.vue' in s)
    chk('...and wraps its %s' % cls,
        re.search(r'<CollapsiblePanel[^>]*class="%s"' % re.escape(cls), s,
                  re.S) is not None)
    chk('...binding a model so the page knows the state',
        'v-model="inputsCollapsed"' in s)
    chk('...and naming it, so the collapsed rail is not anonymous',
        re.search(r'<CollapsiblePanel[^>]*label="[^"]+"', s, re.S) is not None)
    chk('...remembering the choice per page',
        re.search(r'<CollapsiblePanel[^>]*storage-key="[^"]+"', s, re.S)
        is not None)
    # The old wrapper element must be GONE, not left around it: a stray
    # <div class="picker"> would keep the full width and the fold would do
    # nothing visible.
    chk('...with no leftover wrapper of the same class',
        ('<div class="%s">' % cls) not in s
        and ('<aside class="%s">' % cls) not in s)

    if kind == 'grid':
        # THE ONE THAT MATTERS. Without a collapsed track the panel shrinks
        # and the column does not.
        chk('...and the GRID declares a collapsed track',
            'input-collapsed' in s
            and re.search(r'\.input-collapsed\s*{\s*grid-template-columns:\s*30px',
                          s) is not None)
        chk('...bound on the layout element itself',
            "'input-collapsed': inputsCollapsed" in s)

section('The storage keys are distinct')
keys = []
for fname in PAGES:
    p = os.path.join(VIEWS, fname)
    if os.path.exists(p):
        m = re.search(r'storage-key="([^"]+)"', open(p, encoding='utf-8').read())
        if m:
            keys.append(m.group(1))
# Two pages sharing a key would fold each other, which reads as the setting
# leaking between screens rather than as a bug.
chk('no two pages share a remembered key', len(keys) == len(set(keys)),
    str(keys))

print('\n%d passed, %d failed, %d skipped' % (len(OK), len(BAD), len(SKIP)))
if BAD:
    for b in BAD:
        print('  - ' + b)
sys.exit(1 if BAD else 0)
