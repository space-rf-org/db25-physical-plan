#!/usr/bin/env python3
"""The falsifiability gate for db25-physical-plan.

WHY THIS EXISTS. Every increment in this repo has claimed "all mutations killed"
in its commit message, and every one of those claims was a HAND-RUN act: edit the
source, rebuild, watch a test fail, put the source back, write it down. That
makes the claim true on the day and unenforceable thereafter. Nothing stops a
later change from weakening the test that did the killing, and nothing re-runs
the experiment. The umbrella repo has an automated gate (harness/harness.cpp,
`kMutants[]`); this repo had prose.

So: the catalog below IS the record of those experiments, and running this file
re-performs them. A mutation that stops being caught fails CI.

WHAT IT DOES. For each mutant: apply it to the source, rebuild, run ctest, record
which test binaries failed, restore the source. A mutant caught by nobody is a
GATE FAILURE - the behaviour it breaks is no longer defended by any test.

THE RULE THIS TOOL EXISTS TO ENFORCE ON ITSELF. A mutation that does not APPLY
looks exactly like a mutation that was not caught: both show a green suite. That
happened twice while this catalog was being written by hand, and both times the
false result was believed for several minutes. So every mutation asserts its
find-string occurs EXACTLY ONCE before it is applied, and a catalog entry whose
pattern has drifted out of the source is a hard error, never a skip.

WHAT IT DOES NOT DO. It kills at TEST-BINARY granularity, not per test case -
this repo's suites are separate executables with their own main(), not a named
registry the way the umbrella's are. So it detects "this behaviour is defended by
no binary", not "this behaviour is defended by no individual assertion". The
vacuity half of the umbrella's gate (every test killed by some mutant) is
reported below as coverage rather than enforced, because a cost-model catalog
cannot kill the s-expression reader's tests and pretending otherwise would mean
either an allowlist nobody maintains or a gate somebody disables.
"""
import argparse, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# THE CATALOG - the authority, and the record of experiments previously run by
# hand. Each entry names the behaviour it removes; `find` must occur exactly once.
# ---------------------------------------------------------------------------
MUTANTS = [
    dict(id="M1", name="join cardinality reverts to the multiplicative rule (gap register G13)",
         file="src/cost.cpp",
         find="""            const double product = in(0) * in(1);
            // With the keys' distinct counts the containment BOUND becomes the""",
         repl="""            const double product = in(0) * in(1);
            return product * card.join_selectivity;  // MUTANT M1
            // With the keys' distinct counts the containment BOUND becomes the"""),
    dict(id="M2", name="join ignores the keys' distinct counts (always the containment bound)",
         file="src/cost.cpp",
         find="rows = denom > 0.0 ? product / denom : std::max(in(0), in(1));",
         repl="rows = std::max(in(0), in(1));"),
    dict(id="M3", name="the cartesian clamp is removed (a join may exceed |L|x|R|)",
         file="src/cost.cpp",
         find="            return std::min(rows, product);",
         repl="            return rows;"),
    dict(id="M4", name="the first equi-key narrows too (it CONTAINS the join, it does not narrow)",
         file="src/cost.cpp",
         find="            const std::uint32_t contained = join.equi_keys == 0 ? 0u : 1u;",
         repl="            const std::uint32_t contained = 0u;"),
    dict(id="M5", name="a semi join ignores the keys and uses the flat constant",
         file="src/cost.cpp",
         find="            return in(0) * semi_join_fraction(join, card);",
         repl="            return in(0) * card.join_selectivity;"),
    dict(id="M6", name="the anti join is estimated separately from the semi join",
         file="src/cost.cpp",
         find="            return in(0) * (1.0 - semi_join_fraction(join, card));",
         repl="            return in(0) * (1.0 - card.join_selectivity);"),
    dict(id="M7", name="the semi-join fraction is not clamped to one (a subset may exceed its input)",
         file="src/cost.cpp",
         find="    return std::clamp(f, 0.0, 1.0);",
         repl="    return f;"),
    dict(id="M8", name="the Filter ignores the statistics estimate",
         file="src/cost.cpp",
         find="            return in(0) * (filter.selectivity >= 0.0 ? filter.selectivity",
         repl="            return in(0) * (false ? filter.selectivity"),
    dict(id="M9", name="the Aggregate ignores the estimated group count",
         file="src/cost.cpp",
         find="            if (filter.groups > 0.0) return std::min(filter.groups, in(0));",
         repl="            if (false) return std::min(filter.groups, in(0));"),
    dict(id="M10", name="the TREE representation never consults the statistics",
         file="src/cost.cpp",
         find="                         filter_spec_for(node, *this, input_rows.empty() ? 0.0 : input_rows[0]));",
         repl="                         FilterSpec{});"),
    dict(id="M11", name="the MEMO representation never consults the statistics",
         file="src/lowering.cpp",
         find="""                                   filter_spec_of(memo.group(g), cands.front(), card,
                                                  n_in > 0 ? in_rows_buf[0] : 0.0)));""",
         repl="                                   FilterSpec{}));"),
    dict(id="M12", name="equality selectivity ignores the distinct count",
         file="src/selectivity.cpp",
         find="                const double eq = non_null / cs->ndv;",
         repl="                const double eq = defaults.equality;"),
    dict(id="M13", name="range selectivity ignores the column's min/max",
         file="src/selectivity.cpp",
         find="            return clamp_sel(range_selectivity(op, *cs, v) * non_null);",
         repl="            return clamp_sel(defaults.inequality * non_null);"),
    dict(id="M14", name="IS NOT NULL answers IS NULL's number (the negation flag is ignored)",
         file="src/selectivity.cpp",
         find="            return clamp_sel(negated ? 1.0 - nf : nf);",
         repl="            return clamp_sel(nf);"),
    dict(id="M15", name="the group-count product is not capped at the input cardinality",
         file="src/selectivity.cpp",
         find="    return input_rows > 0.0 ? std::min(product, input_rows) : product;",
         repl="    return product;"),
    dict(id="M16", name="one unknown grouping key is skipped rather than refusing the estimate",
         file="src/selectivity.cpp",
         find="        if (cs == nullptr || cs->ndv <= 0.0) return 0.0;",
         repl="        if (cs == nullptr || cs->ndv <= 0.0) continue;"),
    dict(id="M17", name="a computed column (0,0) picks up statistics it must not have",
         file="src/selectivity.cpp",
         find="    if (e.ref_table_id == 0 || e.ref_column_id == 0) return nullptr;",
         repl=""),
    dict(id="M18", name="the fallback is one flat constant rather than split by predicate class",
         file="src/selectivity.cpp",
         find="""                // constant for both is the thing this replaces.
                return equality ? defaults.equality : defaults.inequality;""",
         repl="""                // constant for both is the thing this replaces.
                return defaults.unknown;  // MUTANT M18"""),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, **kw)


def build(builddir):
    return sh(f"cmake --build {builddir} -j4").returncode == 0


def failing_tests(builddir):
    """Names of the ctest targets that FAILED. Empty set == suite is green."""
    r = sh(f"ctest --test-dir {builddir} --output-on-failure")
    out = set()
    for line in r.stdout.splitlines():
        line = line.strip()
        if " - " in line and "(Failed)" in line:
            out.add(line.split(" - ", 1)[1].replace(" (Failed)", "").strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--only", default="", help="run one mutant by id")
    ap.add_argument("--self-test", action="store_true",
                    help="inject a mutation NOTHING can catch and require this gate to fail. "
                         "A gate that has never been seen to fail is not known to be a gate.")
    args = ap.parse_args()
    bd = args.build_dir

    print("=" * 70)
    print(" db25-physical-plan falsifiability gate")
    print("=" * 70)

    # Preflight: verify EVERY catalog pattern resolves before spending a single
    # rebuild. Finding a drifted anchor after fifteen minutes of building teaches
    # the same thing as finding it in two seconds.
    drift = []
    for m in MUTANTS:
        n = (ROOT / m["file"]).read_text().count(m["find"])
        if n != 1:
            drift.append(f"  {m['id']}: pattern occurs {n}x in {m['file']} (expected exactly 1)")
    if drift:
        print("GATE FAIL: the catalog has drifted from the source.")
        print("\n".join(drift))
        print("\nFix the entries. Deleting them silently retires the experiment they record.")
        return 1
    print(f"[preflight] all {len(MUTANTS)} catalog patterns resolve uniquely")

    if not build(bd):
        print("GATE FAIL: the tree does not build clean before any mutation")
        return 1
    clean_failures = failing_tests(bd)
    if clean_failures:
        print(f"GATE FAIL: {len(clean_failures)} test(s) fail CLEAN: {sorted(clean_failures)}")
        return 1
    print(f"[clean] suite is green\n")

    if args.self_test:
        # A change no assertion depends on: a comment. If the gate reports this
        # as CAUGHT, its notion of "caught" is picking up build noise rather than
        # test failure, and every other result it has ever printed is suspect.
        probe = dict(id="PROBE", name="a comment edit, which no test can possibly detect",
                     file="src/cost.cpp",
                     find="// ---- tree form ---------------------------------------------------------",
                     repl="// ---- tree form (self-test probe) ---------------------------------------")
        chosen = [probe]
    else:
        chosen = [m for m in MUTANTS if not args.only or m["id"] == args.only]
    survivors, killers = [], {}

    for m in chosen:
        path = ROOT / m["file"]
        original = path.read_text()
        # A mutation that does not apply is indistinguishable from one that is
        # not caught. Both show green. So this is a hard error, never a skip.
        n = original.count(m["find"])
        if n != 1:
            print(f"GATE FAIL: {m['id']} pattern occurs {n} times in {m['file']} (expected exactly 1).")
            print("           The catalog has drifted from the source. Fix the entry, do not delete it.")
            return 1
        path.write_text(original.replace(m["find"], m["repl"], 1))
        try:
            if not build(bd):
                # A mutation that does not compile has not tested anything.
                print(f"  {m['id']:<4} BUILD FAIL - mutation does not compile; not a result")
                survivors.append(m)
                continue
            failed = failing_tests(bd)
        finally:
            path.write_text(original)
        if failed:
            print(f"  {m['id']:<4} killed by {', '.join(sorted(failed))}")
            for t in failed:
                killers.setdefault(t, []).append(m["id"])
        else:
            print(f"  {m['id']:<4} *** SURVIVED *** {m['name']}")
            survivors.append(m)

    build(bd)  # leave the tree as we found it

    print()
    if args.self_test:
        if survivors:
            print("SELF-TEST PASSED: the probe survived, so this gate can report failure.")
            print("  (the run above shows the GATE FAILED path working)")
            return 0
        print("SELF-TEST FAILED: a comment edit was reported as CAUGHT.")
        print("  This gate cannot distinguish a test failure from noise; do not trust its results.")
        return 1

    if survivors:
        print(f"GATE FAILED: {len(survivors)} mutant(s) survived:")
        for m in survivors:
            print(f"  {m['id']}: {m['name']}")
        return 1

    print(f"GATE PASSED: all {len(chosen)} mutants caught.")
    print("\n[coverage] which binaries did the killing (information, not a gate):")
    for t, ids in sorted(killers.items()):
        print(f"  {t:<22} {len(ids):>2} mutant(s)")
    # Which binaries never did any killing. NOT a gate - see the module docstring -
    # but printed so the hole is visible and shrinking it is a visible improvement
    # rather than an invisible one.
    all_tests = sh(f"ctest --test-dir {bd} -N").stdout
    known = {l.split(":", 1)[1].strip() for l in all_tests.splitlines() if "Test #" in l and ":" in l}
    never = sorted(known - set(killers))
    if never:
        print(f"\n[uncovered] {len(never)} binary/binaries this catalog cannot kill:")
        print("  " + ", ".join(never))
        print("  Not a failure: a cost-model catalog does not reach the s-expr reader.")
        print("  It is the measure of how much of the suite this gate currently defends.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
