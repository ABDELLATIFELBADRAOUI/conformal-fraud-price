# Release checklist

This package is already published. What follows is what was checked, and what
to redo on the next release.

## 1. Verify the released results

```bash
sha256sum -c SHA256SUMS.txt
pip install numpy pandas scipy matplotlib
python src/pipeline.py reproduce
git status --short          # must come back clean
```

`reproduce` rewrites `results/tables/paper_{A,C,D,E,wilcoxon}.csv` and the four
figures from `results/scores/*.npz`. A clean `git status` afterwards is the
check that the committed results and the code that produced them agree.

## 2. Verify no dataset leaked into the repository

```bash
du -sh data/          # README.md only
git log --stat -- data/
```

`.gitignore` excludes the CSVs, but check before every push: a file added by
mistake stays in the history after deletion, and that is a Kaggle licensing
problem, not a tidiness one.

## 3. Environment

`requirements.txt` pins numpy, pandas and xgboost to the versions confirmed in
the notebook's run log. scikit-learn, scipy and matplotlib were not recorded at
run time and are given as floors, not as invented pins. If you re-run the
training path, capture the real versions:

```bash
pip freeze | grep -iE "scikit-learn|scipy|matplotlib"
```

## 4. GitHub

The repository lives at
<https://github.com/ABDELLATIFELBADRAOUI/conformal-fraud-price>. The journal
review is single anonymized, so the repository is public under the authors'
own account and the URL appears directly in the manuscript's Data Availability
statement.

`.gitignore` and `.zenodo.json` are part of the package; a web-upload that
skips dotfiles will drop them.

## 5. Zenodo

1. Connect the GitHub account on zenodo.org and enable this repository.
2. Cut a GitHub release — Zenodo archives it and mints a DOI.
3. Put the DOI in `CITATION.cff` and in the manuscript's Data Availability
   section. The DOI is permanent where a GitHub URL is not.

`.zenodo.json` supplies the title, description, licence, keywords and the four
ORCIDs, so the Zenodo record needs no manual editing.

## 6. After acceptance

Add the article's DOI to `CITATION.cff` under `preferred-citation`, and bump
`version` and `date-released`.
