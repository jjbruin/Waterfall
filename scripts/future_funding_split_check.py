"""Guardrail: how future funding is split at an allocation override (variant D, Oct 7 2026).

Measured against accounting's 12/31/25 tracker (the January board deck's source):
the tracker splits future funding by commitment ratios on every deal but Brainerd,
where INVBPA had funded its whole $5,493,264 commitment, so all $13.3M unfunded was
owed by the TIAA JV -- PSC 10% / TIAA 90%, not the 36.86% PSC of the funded split.
A general "who still owes" walk mis-split Bel Air (PSC1 funded beyond its commitment
there, carrying other investors), so the rule is narrow:

  1. ``group_shares(..., override_weights="remaining")`` weighs an OVERRIDE entity's
     investors by what each still owes there: commitment in force less the override's
     funded amount. Brainerd's real amounts give PSC 10% / TIAA 90% exactly; the
     default (``"funded"``) still gives the funded split, 36.86% / 63.14%.
  2. Nothing changes WITHOUT an override: both weightings give identical shares.
  3. Nothing remaining at the override entity falls back to the funded split, and the
     route says so -- it is never dropped into a default group.
  4. A committed investor the override does not name has funded nothing, so it owes
     its whole commitment.
  5. The PE exposure report uses "remaining" for FUTURE FUNDING only; the funded rows
     keep the funded split.
  6. An unknown weighting is refused.

Fixture: pe_exposure_check's, plus a Brainerd-shaped chain with the real amounts.
Usage: python scripts/future_funding_split_check.py [--inject=ignore|nofallback|unnamed|wiring]
"""
import os
import sys
import tempfile
import types
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pe_exposure_check as base  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=None):
    (_passed if cond else _failed).append(label)
    print(("   ok   " if cond else "   FAIL ") + label + ("" if cond or detail is None else "   [%s]" % (detail,)))


def near(a, b, tol=1e-6):
    return a is not None and abs(float(a) - float(b)) <= tol


D = date(2025, 12, 31)
# Brainerd's chain, with production's commitments in force since 12/27/2024.
base.COMMITMENTS += [
    ("BRNX", "PPIBN", 31721927, "2024-12-27", None),
    ("PPIBN", "INVBN", 16314600, "2024-12-27", None),
    ("PPIBN", "TGAB", 15407327, "2024-12-27", None),
    ("INVBN", "INVAB", 5493264, "2024-12-27", None),
    ("INVBN", "TGAB", 10821336, "2024-12-27", None),
    ("INVAB", "PSC1", 100, "2022-01-01", None),
    ("TGAB", "TGAM", 90, "2022-01-01", None),
    ("TGAB", "PSC1", 10, "2022-01-01", None),
    # Nothing remaining: both investors have funded their whole commitment.
    ("FULLX", "PPIFL", 200, "2022-01-01", None),
    ("PPIFL", "PSC1", 100, "2022-01-01", None),
    ("PPIFL", "TGAM", 100, "2022-01-01", None),
    # A committed investor the override does not name.
    ("UNNX", "PPIUN", 200, "2022-01-01", None),
    ("PPIUN", "PSC1", 100, "2022-01-01", None),
    ("PPIUN", "TGAM", 100, "2022-01-01", None),
]


def oc_module():
    """The ownership engine, with the injected defect when one is asked for."""
    from flask_app.services import ownership_chain_service as oc
    if INJECT not in ("ignore", "nofallback", "unnamed"):
        return oc
    src = open(oc.__file__, encoding="utf-8").read()
    swap = {"ignore": ('if override_weights == "remaining":', "if False:"),
            "nofallback": ("if left > 0.005:", "if True:"),
            "unnamed": ("for i in set(committed) | set(funded)}", "for i in set(funded)}")}[INJECT]
    assert src.count(swap[0]) == 1, "the injection's anchor moved: " + swap[0]
    mod = types.ModuleType("oc_injected")
    mod.__file__ = oc.__file__
    exec(compile(src.replace(*swap), oc.__file__, "exec"), mod.__dict__)
    return mod


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="future_funding_split_check_")
    db = os.path.join(tmp, "check.db")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = db
    eng = base.build_db(db)
    from flask_app import create_app
    app = create_app()
    app.config["DATABASE_URL"] = None
    from flask_app.services import pe_exposure_service as pe
    from flask_app.services import allocation_override_service as aos
    oc = oc_module()

    def shares(holder, inv, weights):
        w = oc.group_shares(holder, D, pe.STOPS, context=pe._context, default_group=pe.DEFAULT_GROUP,
                            engine=eng, investment=inv, override_weights=weights)
        return w

    with app.app_context():
        print("2. Without an override, both weightings are the commitment walk")
        f0, r0 = shares("PPIBN", "BRNX", "funded"), shares("PPIBN", "BRNX", "remaining")
        chk("identical shares with no override", f0["shares"] == r0["shares"], (f0["shares"], r0["shares"]))
        chk("...which is MRI's ratios: PSC 0.5143 x 0.3367 + TGAB x 0.1",
            near(f0["shares"].get("PSC"), (16314600 / 31721927) * (5493264 / 16314600)
                 + ((15407327 / 31721927) + (16314600 / 31721927) * (10821336 / 16314600)) * 0.1),
            f0["shares"])

        print("\n1. Brainerd's overrides: funded split for capital, what is owed for future funding")
        aos.create("PPIBN", "BRNX", "2024-12-27", [{"investor_id": "INVBN", "amount": 9316074.29},
                                                   {"investor_id": "TGAB", "amount": 9091603.11}],
                   "derived from the books", "check", engine=eng)
        aos.create("INVBN", "BRNX", "2024-12-27", [{"investor_id": "INVAB", "amount": 5493264.01},
                                                   {"investor_id": "TGAB", "amount": 3822810.30}],
                   "derived from the books", "check", engine=eng)
        fu, re_ = shares("PPIBN", "BRNX", "funded"), shares("PPIBN", "BRNX", "remaining")
        chk("funded: PSC 36.858% / TIAA 63.142% -- the tracker's funded split",
            near(fu["shares"].get("PSC"), 0.368580, 1e-5) and near(fu["shares"].get("TIAA"), 0.631420, 1e-5),
            fu["shares"])
        chk("remaining: PSC 10% / TIAA 90% -- the tracker's future-funding split",
            near(re_["shares"].get("PSC"), 0.10, 1e-6) and near(re_["shares"].get("TIAA"), 0.90, 1e-6),
            re_["shares"])
        chk("...on Brainerd's $13,314,250 that is PSC $1,331,425 / TIAA $11,982,825",
            round(13314250 * re_["shares"].get("PSC", 0)) == 1331425
            and round(13314250 * re_["shares"].get("TIAA", 0)) == 11982825,
            {g: round(13314250 * v) for g, v in re_["shares"].items()})
        wb = {o["weighed_by"] for rt in re_["routes"] for o in rt.get("overrides") or []}
        chk("every route says it was weighed by what is owed", wb == {"remaining"}, wb)
        other = shares("PPIA", "DEALA", "remaining")
        chk("another deal, with no override, is untouched by the weighting",
            other["shares"] == shares("PPIA", "DEALA", "funded")["shares"], other["shares"])

        print("\n3. Nothing remaining falls back to the funded split, and says so")
        aos.create("PPIFL", "FULLX", "2024-01-01", [{"investor_id": "PSC1", "amount": 100},
                                                    {"investor_id": "TGAM", "amount": 100}],
                   "fully funded", "check", engine=eng)
        fl = shares("PPIFL", "FULLX", "remaining")
        chk("fully funded: PSC 50 / TIAA 50, as funded -- not a default group",
            near(fl["shares"].get("PSC"), 0.5) and near(fl["shares"].get("TIAA"), 0.5)
            and pe.DEFAULT_GROUP not in fl["shares"], fl["shares"])
        wb = {o["weighed_by"] for rt in fl["routes"] for o in rt.get("overrides") or []}
        chk("...and the route says nothing remained", any("nothing remains" in w for w in wb), wb)

        print("\n4. A committed investor the override does not name owes its whole commitment")
        aos.create("PPIUN", "UNNX", "2024-01-01", [{"investor_id": "PSC1", "amount": 100}],
                   "TGAM has not funded", "check", engine=eng)
        un = shares("PPIUN", "UNNX", "remaining")
        chk("PSC1 funded its 100, TGAM owes its 100: future funding is all TIAA",
            near(un["shares"].get("TIAA"), 1.0) and not un["shares"].get("PSC"), un["shares"])

        print("\n5. The report uses it for future funding only")
        src = open(pe.__file__, encoding="utf-8").read()
        if INJECT == "wiring":
            src = src.replace('override_weights="remaining"', 'override_weights="funded"')
        fut = src[src.find("# FUTURE FUNDING"):src.find("future.sort(")]
        rows_part = src[src.find("for h in holdings(inv, commitments, as_of):"):src.find("# FUTURE FUNDING")]
        chk("future funding walks with override_weights=\"remaining\"",
            'override_weights="remaining"' in fut, None)
        chk("the funded rows keep the funded split (no weighting passed)",
            "override_weights" not in rows_part, None)

        print("\n6. An unknown weighting is refused")
        try:
            shares("PPIBN", "BRNX", "owed")
            chk("refused", False)
        except ValueError:
            chk("refused", True)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
