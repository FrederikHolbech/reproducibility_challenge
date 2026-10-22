# Required A01 data

No participant data are committed to this repository. Obtain BCI Competition IV dataset 2a from <https://www.bbci.de/competition/iv/> after completing the site's registration requirements.

Required paths relative to the `Frederik` folder:

```text
data/raw/A01T.gdf
data/raw/A01E.gdf
data/labels/A01T.mat
data/labels/A01E.mat
```

The official label archive is available from <https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip>.

Expected SHA-256 hashes from the completed reproduction:

| File | SHA-256 |
| --- | --- |
| `A01T.gdf` | `74d900ce83a115509f663c0ac45bd36d56f05be528064743c49b0c2efb5088a3` |
| `A01E.gdf` | `08709cf61bc0d603b5d491417221cb4d62a5db09b99ec7bc1c09c22dbc44a674` |
| `A01T.mat` | `b62c9a907eea4d158850e80d7eec5a939d14220cda358a6da78b8897c97b7fa1` |
| `A01E.mat` | `2d2db43b452842553e1690b045b998607131b0b9160ece7989d65632febecc7d` |

Build the normalized cache and evaluate the released checkpoint with:

```powershell
python .\reproduction\evaluate_a01_checkpoint.py --rebuild-cache
```

This creates `data/processed/A01.npz`, which is also excluded from Git.

