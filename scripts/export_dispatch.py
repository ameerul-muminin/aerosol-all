"""Export bilingual dispatch notices with the local forecasting model."""

import argparse
from datetime import date, timedelta
import shlex
import sys
from pathlib import Path

from src.dispatch import brief_html, build_notices, sms, whatsapp
from src.predict import build_features

EXPORTS = Path(__file__).resolve().parent.parent / "exports"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2025-09-15", help="Monday date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--stock-pct", type=float, default=60)
    parser.add_argument("--crews", type=int, default=20)
    parser.add_argument("--beds-pct", type=float, default=70)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    feat_df = build_features()
    try:
        english, bangla = (
            build_notices(feat_df, args.as_of, args.horizon, args.stock_pct, args.crews,
                          args.beds_pct, lang)
            for lang in ("en", "bn")
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Recover the canonical as-of date from the validated CLI input.
    stamp = (date.fromisoformat(english[0]["forecast_week"]) - timedelta(weeks=args.horizon)).isoformat()
    sms_name = f"sms_{stamp}.txt"
    whatsapp_name = f"whatsapp_{stamp}.txt"
    en_name = f"brief_{stamp}_en.html"
    bn_name = f"brief_{stamp}_bn.html"
    command = shlex.join(["python", "-m", "scripts.export_dispatch", *argv])
    readme = f"""# Dispatch exports

- `{sms_name}` contains English and Bangla SMS notices in separate language blocks.
- `{whatsapp_name}` contains English and Bangla multiline WhatsApp notices in separate language blocks.
- `{en_name}` and `{bn_name}` are the English and Bangla printable briefs, with critical wards first.

The date in each filename is the as-of date. Run from the repository root to reproduce these files.

```sh
{command}
```
"""
    files = {
        sms_name: "\n\n\n".join("\n\n".join(map(sms, group)) for group in (english, bangla)) + "\n",
        whatsapp_name: "\n\n\n".join("\n\n".join(map(whatsapp, group)) for group in (english, bangla)) + "\n",
        en_name: brief_html(english),
        bn_name: brief_html(bangla),
        "README.md": readme,
    }
    EXPORTS.mkdir(exist_ok=True)
    for name, content in files.items():
        (EXPORTS / name).write_bytes(content.encode("utf-8"))


if __name__ == "__main__":
    main()
