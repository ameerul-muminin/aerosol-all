# Dispatch exports

- `sms_2025-09-15.txt` contains English and Bangla SMS notices in separate language blocks.
- `whatsapp_2025-09-15.txt` contains English and Bangla multiline WhatsApp notices in separate language blocks.
- `brief_2025-09-15_en.html` and `brief_2025-09-15_bn.html` are the English and Bangla printable briefs, with critical wards first.

Maximum SMS segments per notice: English **1**, Bangla **3**.

The date in each filename is the as-of date. Run from the repository root to reproduce these files.

```sh
python -m scripts.export_dispatch --as-of 2025-09-15 --horizon 2 --stock-pct 60 --crews 20 --beds-pct 70
```
