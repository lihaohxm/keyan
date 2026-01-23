import argparse
import numpy as np
import pandas as pd


def proxy_xi(gamma, m, a=0.6, b=0.4):
    xi = (1 - np.exp(-a * gamma)) * (1 - np.exp(-b * m))
    return np.clip(xi, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="semantic_tables/deepsc_table.csv")
    parser.add_argument("--snr_db_min", type=float, default=-20)
    parser.add_argument("--snr_db_max", type=float, default=30)
    parser.add_argument("--snr_db_step", type=float, default=1)
    parser.add_argument("--m_values", default="4,8,16,32")
    args = parser.parse_args()

    snr_db = np.arange(args.snr_db_min, args.snr_db_max + args.snr_db_step, args.snr_db_step)
    m_values = np.array([int(x) for x in args.m_values.split(",")])

    rows = []
    for m in m_values:
        gamma = 10 ** (snr_db / 10)
        xi = proxy_xi(gamma, m)
        rows.extend(zip(snr_db, np.full_like(snr_db, m, dtype=float), xi))

    df = pd.DataFrame(rows, columns=["snr_db", "M", "xi"])
    df.to_csv(args.out, index=False)
    print(f"Saved semantic table to {args.out}")


if __name__ == "__main__":
    main()
