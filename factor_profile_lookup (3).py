#!/usr/bin/env python
# coding: utf-8

"""
FACTOR PROFILE LOOKUP  (decile inspector)
=========================================
Interactive lookup of a stock's / sector's / sub-sector's factor DECILE profile.

The user is shown a numbered menu of sectors and sub-sectors once at the top,
then repeatedly prompted for input:
    * a NUMBER from the menu   -> that sector / sub-sector
    * a TICKER (no " US")       -> that single stock
    * empty input               -> quit

Deciles are computed ONCE over the FULL universe, per factor (10 = highest
exposure, 1 = lowest). A lookup just READS each name's global decile -- so a
"Quality = 8" means top-20% Quality across ALL stocks, regardless of the sector
queried.

Factors shown (in order):
    structural : Beta, Size
    style      : Quality, SI, GK_Vol, Idio_Mom, Value, OU
    macro      : MACRO_COLS (appended)
Sector / sub-sector dummy columns are ignored.

Output:
    * single stock        -> vertical  factor : decile
    * sector / sub-sector -> DataFrame (stocks down the rows, factors across cols)

Invalid input (unknown ticker / out-of-range number) prints "Invalid selection"
and re-prompts.

DEPENDENCY (Option C): run in the SAME kernel as, and AFTER,
    factor_risk_decomposition.py (reuses its builders + constants) with
    factor_model_step1 loaded. A guard fails loudly if anything's absent.

USAGE
    factor_profile_lookup(Pxs_df, sectors_df, model_version='v2')
"""

import numpy as np
import pandas as pd

# ---- editable globals -------------------------------------------------------
N_DECILES = 10

# structural first, then the rest of the style factors, then macro appended
STRUCTURAL_FACTORS = ['Beta', 'Size']
STYLE_FACTORS      = ['Quality', 'SI', 'GK_Vol', 'Idio_Mom', 'Value', 'OU']


# =============================================================================
# Dependency guard (Option C) — same contract as the screener
# =============================================================================
def _fpl_check_dependencies():
    required = {
        'constants': ['RD_SCALAR_TABLES', 'MACRO_COLS', 'ENGINE'],
        'functions': ['_rd_build_F', '_rd_build_X'],
        'factor_model': ['calc_rolling_betas', 'calc_vol_factor',
                         'calc_macro_betas', 'load_si_composite'],
    }
    g = globals()
    missing = {grp: [n for n in names if n not in g]
               for grp, names in required.items()}
    missing = {grp: ns for grp, ns in missing.items() if ns}
    if missing:
        lines = ["Missing dependencies — run factor_risk_decomposition.py first "
                 "in the SAME kernel (and load factor_model_step1)."]
        for grp, ns in missing.items():
            lines.append(f"   missing {grp}: {', '.join(ns)}")
        raise NameError("\n".join(lines))


# =============================================================================
# Full-universe construction (mirror of the screener / optimizer)
# =============================================================================
def _fpl_build_full_universe(Pxs_df, sectors_s):
    extended_st_dt = Pxs_df.index[0]
    try:
        from sqlalchemy import text as _text
        with ENGINE.connect() as conn:
            rows = conn.execute(_text(
                "SELECT DISTINCT ticker FROM income_data")).fetchall()
        db_tickers = {r[0].upper() for r in rows}
    except Exception:
        db_tickers = set(sectors_s.index)
    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]
    universe = []
    for col in sectors_s.index:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        if col not in Pxs_df.columns:
            continue
        if len(pre_dates) >= 252:
            cd = Pxs_df.loc[pre_dates[-252:], col]
            if isinstance(cd, pd.DataFrame):
                cd = cd.iloc[:, 0]
            if int(cd.notna().sum()) < 126:
                continue
        universe.append(col)
    return universe


# =============================================================================
# Deciles (over the full universe)
# =============================================================================
def _to_deciles(series):
    """Rank-based deciles 1..10 (10 = highest). NaNs stay NaN."""
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float, index=series.index)
    r = s.rank(method='average', pct=True)
    dec = np.ceil(r * N_DECILES).clip(1, N_DECILES).astype(int)
    return dec.reindex(series.index)


# =============================================================================
# Entry point
# =============================================================================
def factor_profile_lookup(Pxs_df, sectors_df, volumeTrd_df=None,
                          port_s=None, model_version='v2'):
    """Interactive decile-profile inspector. See module docstring."""
    _fpl_check_dependencies()

    # resolve sector inputs — detect the sector / sub-sector columns flexibly, so
    # the script reads whatever classification the caller passes (not a fixed name).
    if isinstance(sectors_df, pd.DataFrame):
        cols_lower = {c.lower().replace('-', '_').replace(' ', '_'): c
                      for c in sectors_df.columns}
        # sector column: first match among common spellings
        _sec_key = next((cols_lower[k] for k in
                         ('sector', 'sectors', 'gics_sector') if k in cols_lower), None)
        if _sec_key is None:
            _sec_key = sectors_df.columns[0]   # fall back to the first column
        sectors_s = sectors_df[_sec_key].copy()
        # sub-sector column: match sub_sector / subsector / sub-sector / etc.
        _sub_key = next((cols_lower[k] for k in
                         ('sub_sector', 'subsector', 'sub_sectors', 'subsectors',
                          'gics_subsector') if k in cols_lower), None)
        subsec_s = sectors_df[_sub_key].copy() if _sub_key is not None else None
        print(f"  sector column = '{_sec_key}'"
              + (f"   sub-sector column = '{_sub_key}'" if _sub_key
                 else "   (no sub-sector column found)"))
    else:
        sectors_s = sectors_df.copy()
        subsec_s  = None
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    if subsec_s is not None:
        subsec_s = subsec_s[~subsec_s.index.duplicated(keep='first')]

    # build full universe + factor model exposures
    print("[1/2] Building factor model over the full universe...")
    F, factor_names, sec_cols, subsec_cols = _rd_build_F(model_version=model_version)
    full_universe = _fpl_build_full_universe(Pxs_df, sectors_s)
    print(f"  Full universe: {len(full_universe)} stocks")
    X_full = _rd_build_X(full_universe, factor_names, sec_cols, subsec_cols,
                         Pxs_df, sectors_s, subsec_s, volumeTrd_df,
                         model_version=model_version)

    # factor list to display: structural + style + macro (present in the model),
    # ignoring sector/sub-sector dummy columns.
    macro = [f for f in MACRO_COLS if f in factor_names]
    ordered_factors = [f for f in (STRUCTURAL_FACTORS + STYLE_FACTORS + macro)
                       if f in factor_names
                       and f not in sec_cols and f not in subsec_cols]

    # compute global deciles ONCE over the full universe, per factor
    print("[2/2] Quantizing full universe into deciles per factor...")
    dec_full = pd.DataFrame(index=X_full.index)
    for f in ordered_factors:
        dec_full[f] = _to_deciles(X_full[f])

    # -- build the numbered sector / sub-sector menu (printed once) ----------
    # IMPORTANT: build the menu from the CALLER's classification (sectors_s /
    # subsec_s), restricted to stocks actually in the decile universe -- so every
    # menu entry is guaranteed to have members, and the names match the lookup
    # source exactly (avoids a menu built from the model's dummy columns whose
    # names may differ from the caller's sub-sector labels).
    _uni = dec_full.index
    _sec_in_uni = sectors_s.reindex(_uni).dropna()
    sec_list = sorted(pd.Index(_sec_in_uni.unique()))
    if subsec_s is not None:
        _sub_in_uni = subsec_s.reindex(_uni).dropna()
        subsec_list = sorted(pd.Index(_sub_in_uni.unique()))
    else:
        subsec_list = []
    # menu maps a display number -> (kind, name)
    menu = {}
    n = 0

    # -- normalise the portfolio (if provided) over IN-UNIVERSE constituents --
    _port_w = None
    if port_s is not None and len(port_s) > 0:
        _p = pd.Series(port_s).dropna()
        _p = _p[~_p.index.duplicated(keep='first')]
        _in_uni = [t for t in _p.index if t in dec_full.index]
        _missing_port = [t for t in _p.index if t not in dec_full.index]
        _w = _p.reindex(_in_uni).astype(float)
        _tot = _w.sum()
        if _tot != 0 and len(_in_uni) > 0:
            _port_w = _w / _tot        # weights renormalised across in-universe names
        # PORTFOLIO is menu option 1 (before sectors)
        print("\n" + "=" * 66)
        print("  PORTFOLIO")
        print("=" * 66)
        n += 1; menu[n] = ('portfolio', 'PORTFOLIO')
        print(f"  {n:>3}. PORTFOLIO  ({len(_in_uni)} in-universe constituent(s)"
              + (f", {len(_missing_port)} not in universe" if _missing_port else "")
              + ")")

    print("\n" + "=" * 66)
    print("  SECTORS")
    print("=" * 66)
    for s in sec_list:
        n += 1; menu[n] = ('sector', s)
        print(f"  {n:>3}. {s}")
    if subsec_list:
        print("\n" + "=" * 66)
        print("  SUB-SECTORS")
        print("=" * 66)
        for s in subsec_list:
            n += 1; menu[n] = ('subsector', s)
            print(f"  {n:>3}. {s}")
    print("=" * 66)
    print("  Enter a NUMBER (sector/sub-sector), a TICKER, or empty to quit.\n")

    # ticker lookup set (upper-cased, no ' US')
    _tick_norm = {t.upper(): t for t in dec_full.index}

    # -- interactive loop ----------------------------------------------------
    while True:
        raw = input("  stock ticker / sector # (empty = quit): ").strip()
        if raw == "":
            print("  Done.")
            break

        # NUMBER -> portfolio / sector / sub-sector
        if raw.lstrip('+-').isdigit():
            idx = int(raw)
            if idx not in menu:
                print("  Invalid selection")
                continue
            kind, name = menu[idx]
            if kind == 'portfolio':
                if _port_w is None or len(_port_w) == 0:
                    print("  Invalid selection")   # no in-universe constituents
                    continue
                _print_portfolio_profile(_port_w, dec_full, ordered_factors,
                                         sectors_s, subsec_s,
                                         missing=_missing_port)
                continue
            src = sectors_s if kind == 'sector' else (subsec_s if subsec_s is not None
                                                      else sectors_s)
            members = [t for t in dec_full.index if src.get(t) == name]
            if not members:
                print("  Invalid selection")   # nothing eligible in that bucket
                continue
            _print_sector_profile(name, members, dec_full, ordered_factors)
            continue

        # TICKER -> single stock (accept with/without ' US', case-insensitive)
        key = raw.upper().replace(' US', '').strip()
        tk = _tick_norm.get(key) or _tick_norm.get(key + ' US')
        if tk is None or tk not in dec_full.index:
            print("  Invalid selection")
            continue
        _print_stock_profile(tk, sectors_s, subsec_s, dec_full, ordered_factors)


# =============================================================================
# Printing
# =============================================================================
def _print_stock_profile(tk, sectors_s, subsec_s, dec_full, factors):
    sec = sectors_s.get(tk, '')
    sub = subsec_s.get(tk, '') if subsec_s is not None else ''
    print("\n" + "=" * 46)
    print(f"  {tk}   factor deciles (1=low .. 10=high)")
    hdr = f"  sector: {sec}"
    if sub:
        hdr += f"   |   sub-sector: {sub}"
    print(hdr)
    print("=" * 46)
    for f in factors:
        v = dec_full.at[tk, f] if (tk in dec_full.index) else np.nan
        vs = f"{int(v):>2d}" if pd.notna(v) else " n/a"
        print(f"  {f:<12} {vs}")
    print("=" * 46 + "\n")


def _print_portfolio_profile(port_w, dec_full, factors, sectors_s, subsec_s,
                             missing=None):
    """Portfolio view: each in-universe constituent's factor deciles (weight%,
    integer deciles, '-' for missing), then a composite row = allocation-weighted
    average of the deciles per factor (1 decimal). The weighted average is
    renormalised PER FACTOR over the constituents that have a valid decile for
    that factor (missing deciles don't drag the composite toward any value)."""
    names = list(port_w.index)
    # order constituents by weight (desc) for readability
    order = list(port_w.sort_values(ascending=False).index)

    print("\n" + "=" * 90)
    print(f"  PORTFOLIO   —   {len(names)} in-universe constituent(s)   "
          f"factor deciles (1=low .. 10=high)")
    print("=" * 90)

    # constituent table: Weight% + integer-string deciles
    disp = pd.DataFrame(index=order)
    disp['Wgt%'] = [f"{port_w[t]*100:.1f}" for t in order]
    for f in factors:
        disp[f] = [(f"{int(dec_full.at[t, f])}"
                    if (t in dec_full.index and pd.notna(dec_full.at[t, f]))
                    else "-") for t in order]
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 220):
        print(disp.to_string())

    # composite row: per-factor weighted mean of deciles over valid names,
    # weights renormalised across the names that have that factor's decile.
    comp = {}
    for f in factors:
        vals = pd.Series(
            {t: dec_full.at[t, f] for t in order
             if (t in dec_full.index and pd.notna(dec_full.at[t, f]))},
            dtype=float)
        if len(vals) == 0:
            comp[f] = np.nan
            continue
        w = port_w.reindex(vals.index).astype(float)
        wsum = w.sum()
        comp[f] = float((w * vals).sum() / wsum) if wsum != 0 else np.nan

    print("  " + "-" * 86)
    comp_cells = "  ".join(
        f"{f[:8]}={comp[f]:.1f}" if pd.notna(comp[f]) else f"{f[:8]}=-"
        for f in factors)
    print(f"  COMPOSITE (allocation-weighted avg of deciles):")
    print(f"    {comp_cells}")
    if missing:
        print(f"  Excluded {len(missing)} constituent(s) not in universe: "
              f"{list(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    print("=" * 90 + "\n")


def _print_sector_profile(name, members, dec_full, factors):
    sub = dec_full.loc[members, factors].copy()
    # order rows by the first factor's decile desc, then ticker, for readability
    sub = sub.sort_values(by=factors[0], ascending=False)
    print("\n" + "=" * 78)
    print(f"  {name}   —   {len(members)} stock(s)   factor deciles (1=low .. 10=high)")
    print("=" * 78)
    # Format every decile as a clean integer STRING ("10"), with a placeholder
    # for NaN -- so a NaN anywhere in a column can't upcast it to float and
    # introduce ".0" decimals. String cells => no decimals, ever.
    disp = pd.DataFrame(index=sub.index)
    for f in factors:
        disp[f] = sub[f].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 200):
        print(disp.to_string())
    print("=" * 78 + "\n")


if __name__ == "__main__":
    print(__doc__)
