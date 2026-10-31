# External model weight

The final weight is not committed to Git. Download both parts linked in
[`weights_download.json`](../weights_download.json), then run from the repository root:

```powershell
python SystemCode/model/endolens_e13/install_weights.py --parts-dir "$HOME/Downloads"
```

The helper verifies each part, concatenates them without conversion and verifies
the original 130,413,723-byte file. It refuses an incorrect existing destination.
See [the release guide](../README.md) for links and SDK verification.

The installer defaults to `--target both`: it provisions the existing backend and
the standalone SDK without modifying backend code. Use `--target backend` or
`--target sdk` to provision only one location. Two copies require about 249 MiB.
