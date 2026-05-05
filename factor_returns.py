"""
factor_returns.py
─────────────────
Query and display factor returns from v2 lambda tables.

Usage:
    exec(open('factor_returns.py').read())
"""

from sqlalchemy import create_engine, text
import pandas as pd

ENGINE = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")

LAMBDA_TABLES = {
    'Market Beta'   : 'v2_lambda_mkt',
    'Quality'       : 'v2_lambda_quality',
    'Idio Momentum' : 'v2_lambda_mom',
    'Size'          : 'v2_lambda_size',
    'Value'         : 'v2_lambda_value',
    'SI Composite'  : 'v2_lambda_si',
    'GK Vol'        : 'v2_lambda_vol',
    'Macro'         : 'v2_lambda_macro',
    'Sectors'       : 'v2_lambda_sec',
    'O-U Reversion' : 'v2_lambda_ou',
}

def load_factor_returns(start_date=None, end_date=None):
    """
    Load factor returns from v2 lambda tables.
    Returns DataFrame (dates x factors), each cell = daily factor return.

    Parameters
    ----------
    start_date : str or pd.Timestamp, optional  e.g. '2019-01-01'
    end_date   : str or pd.Timestamp, optional  e.g. '2026-05-01'
    """
    series = {}

    for factor_name, table in LAMBDA_TABLES.items():
        try:
            # Introspect column names
            with ENGINE.connect() as conn:
                cols = conn.execute(text(f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """)).fetchall()
            col_names = [c[0] for c in cols]

            # Value columns = everything except date, r2, ridge_lambda, index
            skip = {'date', 'r2', 'ridge_lambda', 'index', 'intercept'}
            val_cols = [c for c in col_names if c not in skip]

            if not val_cols:
                print(f"  WARNING: no value columns in {table} (cols: {col_names})")
                continue

            # Quote all column names to handle reserved words
            val_cols_sql = ', '.join([f'"{c}"' for c in val_cols])
            query = f'SELECT date, {val_cols_sql} FROM {table}'
            filters = []
            if start_date:
                filters.append(f"date >= '{pd.Timestamp(start_date).date()}'")
            if end_date:
                filters.append(f"date <= '{pd.Timestamp(end_date).date()}'")
            if filters:
                query += " WHERE " + " AND ".join(filters)
            query += " ORDER BY date"

            with ENGINE.connect() as conn:
                df = pd.read_sql(text(query), conn)

            if df.empty:
                print(f"  WARNING: no data in {table}")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Sum across all factor columns for total step return
            series[factor_name] = df[val_cols].astype(float).sum(axis=1)

        except Exception as e:
            print(f"  WARNING: could not load {table}: {e}")

    if not series:
        print("No factor returns loaded.")
        return pd.DataFrame()

    result = pd.DataFrame(series).sort_index()
    return result


def display_factor_returns(df, tail_n=20):
    """Print summary statistics and recent returns."""
    if df.empty:
        print("No data to display.")
        return

    print("=" * 70)
    print(f"  FACTOR RETURNS SUMMARY  |  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  {len(df)} trading days")
    print("=" * 70)

    # Summary stats
    print(f"\n{'Factor':<18} {'Ann. Ret':>9} {'Ann. Vol':>9} {'Sharpe':>8} {'Min':>9} {'Max':>9}")
    print("-" * 66)
    for col in df.columns:
        s       = df[col].dropna()
        ann_ret = float(s.mean() * 252 * 100)
        ann_vol = float(s.std() * (252 ** 0.5) * 100)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else float('nan')
        print(f"  {col:<16} {ann_ret:>+8.2f}%  {ann_vol:>8.2f}%  "
              f"{sharpe:>7.2f}  {float(s.min()*100):>+8.2f}%  {float(s.max()*100):>+8.2f}%")

    # Recent returns
    print(f"\n  LAST {tail_n} TRADING DAYS\n")
    recent = df.tail(tail_n) * 100
    recent.index = recent.index.strftime('%Y-%m-%d')
    print(recent.to_string(float_format=lambda x: f"{x:+.2f}%"))
    print()


def load_sector_returns(start_date=None, end_date=None):
    """Load individual sector lambda returns from v2_lambda_sec."""
    table = 'v2_lambda_sec'
    try:
        with ENGINE.connect() as conn:
            cols = conn.execute(text(f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)).fetchall()
        col_names = [c[0] for c in cols]
        skip = {'date', 'r2', 'ridge_lambda', 'index', 'intercept'}
        val_cols = [c for c in col_names if c not in skip]

        val_cols_sql = ', '.join([f'"{c}"' for c in val_cols])
        query = f'SELECT date, {val_cols_sql} FROM {table}'
        filters = []
        if start_date:
            filters.append(f"date >= '{pd.Timestamp(start_date).date()}'")
        if end_date:
            filters.append(f"date <= '{pd.Timestamp(end_date).date()}'")
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY date"

        with ENGINE.connect() as conn:
            df = pd.read_sql(text(query), conn)

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[val_cols].astype(float)
        return df

    except Exception as e:
        print(f"  WARNING: could not load sector returns: {e}")
        return pd.DataFrame()


def display_sector_returns(df, tail_n=10):
    """Print per-sector summary stats and recent returns."""
    if df.empty:
        print("No sector data to display.")
        return

    print("=" * 70)
    print(f"  SECTOR RETURNS  |  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  {len(df)} trading days")
    print("=" * 70)

    print(f"\n{'Sector':<35} {'Ann. Ret':>9} {'Ann. Vol':>9} {'Sharpe':>8} {'Min':>9} {'Max':>9}")
    print("-" * 80)
    rows = []
    for col in df.columns:
        s       = df[col].dropna()
        ann_ret = float(s.mean() * 252 * 100)
        ann_vol = float(s.std() * (252 ** 0.5) * 100)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else float('nan')
        rows.append((ann_ret, col, ann_vol, sharpe, s.min()*100, s.max()*100))

    # Sort by annualized return descending
    for ann_ret, col, ann_vol, sharpe, mn, mx in sorted(rows, reverse=True):
        print(f"  {col:<33} {ann_ret:>+8.2f}%  {ann_vol:>8.2f}%  "
              f"{sharpe:>7.2f}  {mn:>+8.2f}%  {mx:>+8.2f}%")

    print(f"\n  LAST {tail_n} TRADING DAYS\n")
    recent = df.tail(tail_n) * 100
    recent.index = recent.index.strftime('%Y-%m-%d')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(recent.to_string(float_format=lambda x: f"{x:+.2f}%"))
    print()


# ── Run on exec ───────────────────────────────────────────────────────────────
factor_returns_df = load_factor_returns()
display_factor_returns(factor_returns_df)

sector_returns_df = load_sector_returns()
display_sector_returns(sector_returns_df)
