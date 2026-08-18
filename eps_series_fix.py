#!/usr/bin/env python
# coding: utf-8

"""
EPS-series fix  (interactive, reversible — writes to income_data)
=================================================================
Generic repair for the 'eps' (GAAP basic) and 'dilutedEps' (GAAP diluted)
series, applied to a LIST of stocks. One selected mode is applied to every
ticker in the input list, across ALL download_date vintages.

MODES
  1. eps  -> dilutedEps           (copy full eps series into dilutedEps)
  2. dilutedEps -> eps            (copy full dilutedEps series into eps)
  3. SPLIT AT FEQ (first estimated quarter = FEP):
       period >= FEQ (FEQ included) :  eps <- dilutedEps
       period <  FEQ (FEQ excluded) :  dilutedEps <- eps
     i.e. diluted is authoritative from the first estimated quarter onward;
     basic is authoritative for the historical (actual) quarters before it.
     This is usually the right choice: actual quarters anchor on Ortex basic
     eps, forward/estimated quarters anchor on the cleaner diluted series.

RULES
  - ALL vintages: every (period, download_date) row of the source is copied
    onto the matching target row (same ticker/period/download_date).
  - SKIP ON MISSING SOURCE: if the source value is NULL/NaN for a cell, the
    target is left untouched (never destroy a real value with a missing one).
  - If the target row does not exist for a (period, download_date) that the
    source has, a new target row is INSERTED (so the copy is complete).
  - Backup + dry-run + revert, mirroring the split script:
      * dry_run=True (default) previews, writes nothing.
      * every changed/inserted cell is backed up under label 'EPS_FIX' before
        the write; revert('TICKER') restores them.

USAGE
    from eps_series_fix import fix_eps_series, revert
    fix_eps_series(['LITE','COHR'])                         # prompts for mode, dry-run
    fix_eps_series(['LITE','COHR'], mode=3)                 # mode given, dry-run
    fix_eps_series(['LITE','COHR'], mode=3, dry_run=False)  # apply
    revert('LITE')                                          # undo EPS_FIX on LITE
"""

import math
import pandas as pd
from sqlalchemy import create_engine, text

CONNECTION_STRING = "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
ENGINE = create_engine(CONNECTION_STRING)

TABLE          = 'income_data'
CATEGORY       = 'income'                    # estimation_status category
BASIC_METRIC   = 'eps'
DILUTED_METRIC = 'dilutedEps'
BACKUP_TBL     = 'eps_fix_backup'
LABEL          = 'EPS_FIX'


# ----------------------------------------------------------------------------
def _qkey(period: str):
    """'2026Q4' -> (2026, 4) for correct chronological ordering."""
    try:
        y, q = period.upper().split('Q')
        return (int(y), int(q))
    except (ValueError, AttributeError):
        return (9999, 9)          # unparseable -> sort last, treated as >= any FEQ


def get_fep(ticker):
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period FROM estimation_status
            WHERE ticker = :t AND category = :c
        """), {"t": ticker, "c": CATEGORY}).fetchone()
    return row[0] if row and row[0] else None


def _ensure_backup_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {BACKUP_TBL} (
                backup_stamp   TEXT,
                ticker         TEXT,
                period         TEXT,
                download_date  TIMESTAMP,
                metric_name    TEXT,
                old_value      DOUBLE PRECISION,   -- NULL = row was INSERTED (no prior)
                new_value      DOUBLE PRECISION,
                est_flag       BOOLEAN,
                label          TEXT
            )
        """))


def _is_finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
def _plan_for_ticker(ticker: str, mode: int):
    """Return (changes, inserts) for one ticker under the selected mode.
      changes: (period, dd, target_metric, old_val, new_val, est)
      inserts: (period, dd, target_metric, new_val, est)   [target row absent]
    Skips cells whose SOURCE value is missing.
    """
    t = ticker.strip().upper()
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT period, download_date, metric_name, value, estimated_values
            FROM {TABLE}
            WHERE ticker = :t AND metric_name IN (:mb, :md)
        """), {"t": t, "mb": BASIC_METRIC, "md": DILUTED_METRIC}).fetchall()
    if not rows:
        return [], [], None

    # index by (period, download_date) -> {metric: (value, est)}
    cells = {}
    for period, dd, m, v, est in rows:
        cells.setdefault((period, dd), {})[m] = (v, est)

    fep = None
    if mode == 3:
        fep = get_fep(t)
        if fep is None:
            return [], [], 'NO_FEP'          # cannot split without FEP
        fep_k = _qkey(fep)

    def _direction(period):
        """Return (source_metric, target_metric) for this period under the mode."""
        if mode == 1:
            return BASIC_METRIC, DILUTED_METRIC
        if mode == 2:
            return DILUTED_METRIC, BASIC_METRIC
        # mode 3: period >= FEQ -> diluted->eps ; period < FEQ -> eps->diluted
        if _qkey(period) >= fep_k:
            return DILUTED_METRIC, BASIC_METRIC
        return BASIC_METRIC, DILUTED_METRIC

    changes, inserts = [], []
    for (period, dd), mv in cells.items():
        src_m, tgt_m = _direction(period)
        src = mv.get(src_m)
        if src is None or not _is_finite(src[0]):
            continue                          # skip: missing source
        src_val, src_est = float(src[0]), src[1]
        tgt = mv.get(tgt_m)
        if tgt is None:
            # target row absent -> insert (inherit source's est flag)
            inserts.append((period, dd, tgt_m, src_val, src_est))
        else:
            old_val = tgt[0]
            old_f = float(old_val) if _is_finite(old_val) else None
            if old_f is None or abs(src_val - old_f) > 1e-12:
                changes.append((period, dd, tgt_m, old_f, src_val, tgt[1]))
    return changes, inserts, fep


# ----------------------------------------------------------------------------
def _prompt_mode():
    """Interactive mode selection. Returns 1, 2, 3, or None (aborted)."""
    print("\n" + "=" * 70)
    print("  SELECT EPS FIX MODE")
    print("=" * 70)
    print("  1. eps -> dilutedEps                (copy basic into diluted, all periods)")
    print("  2. dilutedEps -> eps                (copy diluted into basic, all periods)")
    print("  3. SPLIT AT FEQ (first estimated quarter):")
    print("       period >= FEQ : dilutedEps -> eps   (diluted authoritative)")
    print("       period <  FEQ : eps -> dilutedEps   (basic authoritative)")
    print("=" * 70)
    while True:
        raw = input("  Select mode (1/2/3, Enter=cancel): ").strip()
        if raw == "":
            print("  cancelled."); return None
        if raw in ("1", "2", "3"):
            return int(raw)
        print("  invalid selection.")


def fix_eps_series(tickers, mode: int = None, dry_run: bool = True):
    """Apply one EPS-series copy mode to every ticker in `tickers`.

    mode : 1 = eps->dilutedEps | 2 = dilutedEps->eps | 3 = split at FEQ.
           If None (default), prompt for it at runtime.
    dry_run : preview only (default). Set False to write.
    """
    if mode is None:
        mode = _prompt_mode()
        if mode is None:
            return
    if mode not in (1, 2, 3):
        print("  mode must be 1, 2, or 3."); return
    if isinstance(tickers, str):
        tickers = [tickers]
    _ensure_backup_table()

    mode_desc = {
        1: "eps -> dilutedEps (all periods)",
        2: "dilutedEps -> eps (all periods)",
        3: "SPLIT@FEQ: >=FEQ diluted->eps ; <FEQ eps->diluted",
    }[mode]
    print("=" * 70)
    print(f"  EPS SERIES FIX  |  mode {mode}: {mode_desc}")
    print(f"  stocks: {len(tickers)}   dry_run={dry_run}")
    print("=" * 70)

    grand_changes = grand_inserts = 0
    per_ticker = {}
    for tk in tickers:
        t = tk.strip().upper()
        changes, inserts, fep = _plan_for_ticker(t, mode)
        if fep == 'NO_FEP':
            print(f"  {t:<8} SKIPPED — no FEP on record (needed for mode 3).")
            continue
        per_ticker[t] = (changes, inserts)
        n_c, n_i = len(changes), len(inserts)
        grand_changes += n_c; grand_inserts += n_i
        fep_s = f"  FEQ={fep}" if mode == 3 else ""
        print(f"  {t:<8} {n_c:>4} update(s), {n_i:>3} insert(s){fep_s}")
        # small sample
        for period, dd, m, old, new, est in changes[:2]:
            o = f"{old:.4f}" if old is not None else "NULL"
            print(f"      {m:<11} {period} [{str(dd)[:10]}]  {o:>10} -> {new:>10.4f}")
        for period, dd, m, new, est in inserts[:1]:
            print(f"      {m:<11} {period} [{str(dd)[:10]}]  (insert) -> {new:>10.4f}")

    print("-" * 70)
    print(f"  TOTAL: {grand_changes} update(s), {grand_inserts} insert(s) "
          f"across {len(per_ticker)} stock(s)")

    if dry_run:
        print("  DRY RUN — nothing written. Call with dry_run=False to apply.")
        print("=" * 70)
        return
    if grand_changes + grand_inserts == 0:
        print("  Nothing to change."); print("=" * 70); return
    if input(f"  Apply mode {mode} to {len(per_ticker)} stock(s)? (y/n): ").strip().lower() != 'y':
        print("  aborted."); return

    stamp = f"{LABEL}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    total_u = total_i = 0
    with ENGINE.begin() as conn:
        for t, (changes, inserts) in per_ticker.items():
            # updates
            for period, dd, m, old, new, est in changes:
                conn.execute(text(f"""
                    INSERT INTO {BACKUP_TBL}
                      (backup_stamp,ticker,period,download_date,metric_name,
                       old_value,new_value,est_flag,label)
                    VALUES (:s,:t,:p,:dd,:m,:ov,:nv,:e,:lab)
                """), {"s":stamp,"t":t,"p":period,"dd":dd,"m":m,
                       "ov":old,"nv":new,"e":est,"lab":LABEL})
                conn.execute(text(f"""
                    UPDATE {TABLE} SET value = :v
                    WHERE ticker=:t AND period=:p AND download_date=:dd
                      AND metric_name=:m
                """), {"v":new,"t":t,"p":period,"dd":dd,"m":m})
                total_u += 1
            # inserts (old_value NULL in backup = "was inserted" -> revert deletes)
            for period, dd, m, new, est in inserts:
                conn.execute(text(f"""
                    INSERT INTO {BACKUP_TBL}
                      (backup_stamp,ticker,period,download_date,metric_name,
                       old_value,new_value,est_flag,label)
                    VALUES (:s,:t,:p,:dd,:m,NULL,:nv,:e,:lab)
                """), {"s":stamp,"t":t,"p":period,"dd":dd,"m":m,
                       "nv":new,"e":est,"lab":LABEL})
                conn.execute(text(f"""
                    INSERT INTO {TABLE}
                      (ticker,period,download_date,metric_name,value,estimated_values)
                    VALUES (:t,:p,:dd,:m,:v,:e)
                """), {"t":t,"p":period,"dd":dd,"m":m,"v":new,"e":est})
                total_i += 1

    print(f"  Applied: {total_u} updated, {total_i} inserted. "
          f"revert('<ticker>') to undo (stamp {stamp}).")
    print("=" * 70)


# ----------------------------------------------------------------------------
def revert(ticker: str, label: str = LABEL):
    """Undo the most recent EPS_FIX on a ticker: restore old_value for updates,
    DELETE rows that were inserted (old_value IS NULL in backup)."""
    t = ticker.strip().upper()
    with ENGINE.connect() as conn:
        # most recent stamp for this ticker/label
        stamp_row = conn.execute(text(f"""
            SELECT backup_stamp FROM {BACKUP_TBL}
            WHERE ticker=:t AND label=:lab
            ORDER BY backup_stamp DESC LIMIT 1
        """), {"t":t,"lab":label}).fetchone()
        if not stamp_row:
            print(f"  No {label} backup on record for {t}."); return
        stamp = stamp_row[0]
        rows = conn.execute(text(f"""
            SELECT period, download_date, metric_name, old_value
            FROM {BACKUP_TBL}
            WHERE ticker=:t AND label=:lab AND backup_stamp=:s
        """), {"t":t,"lab":label,"s":stamp}).fetchall()
    if not rows:
        print(f"  Nothing to revert for {t} (stamp {stamp})."); return

    restored = deleted = 0
    with ENGINE.begin() as conn:
        for period, dd, m, old in rows:
            if old is None:
                # was an INSERT -> delete it
                conn.execute(text(f"""
                    DELETE FROM {TABLE}
                    WHERE ticker=:t AND period=:p AND download_date=:dd
                      AND metric_name=:m
                """), {"t":t,"p":period,"dd":dd,"m":m})
                deleted += 1
            else:
                conn.execute(text(f"""
                    UPDATE {TABLE} SET value=:v
                    WHERE ticker=:t AND period=:p AND download_date=:dd
                      AND metric_name=:m
                """), {"v":old,"t":t,"p":period,"dd":dd,"m":m})
                restored += 1
    print(f"  Reverted {t} (stamp {stamp}): {restored} restored, {deleted} deleted.")


if __name__ == "__main__":
    print(__doc__)
