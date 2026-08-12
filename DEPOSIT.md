# Avant de déposer — liste de contrôle

## 1. Résultats — déjà inclus dans ce paquet

Les 13 `.npz`, les tables `paper_*` + `table02_*`, `all_results.json`,
`PAPER_RESULTS.md` et les quatre figures sont en place et vérifiés : chaque
cellule du Tableau 3 du papier a été recalculée depuis les `.npz` et coïncide.

```bash
sha256sum -c SHA256SUMS.txt
```

## 2. Compléter les champs marqués

| Fichier | À remplir |
|---|---|
| `CITATION.cff` | `repository-code`, et les ORCID si vous en avez |
| `data/README.md` | les quatre SHA-256 |
| le manuscrit | le jeton `[TO COMPLETE: repository URL]` dans Data Availability |

## 2bis. Figer l'environnement

Sur la machine qui a produit les résultats :

```bash
pip freeze | grep -iE "scikit-learn|scipy|matplotlib"
```

et reporter les trois versions dans `requirements.txt` (numpy, pandas et
xgboost y sont déjà, confirmés par le journal d'exécution du notebook).

## 3. Vérifier qu'aucune donnée n'est incluse

```bash
git status --short
du -sh data/
```

`data/` ne doit contenir que `README.md`. Le `.gitignore` exclut les CSV, mais
vérifiez avant le premier commit : un fichier ajouté par erreur reste dans
l'historique même après suppression, et c'est alors un problème de licence
Kaggle.

## 4. Vérifier que le notebook tourne depuis les scores seuls

Avant de publier, redémarrez le noyau et exécutez uniquement les cellules 1, 2,
puis 12, 13 et 14. Elles ne lisent que les `.npz`. Si elles produisent
`PAPER_RESULTS.md` et les quatre figures sans toucher aux données brutes, la
promesse du README tient.

## 5. GitHub

```bash
git init
git add .
git commit -m "Reproduction package for the conformal coverage price paper"
git branch -M main
git remote add origin https://github.com/[USER]/conformal-fraud-price.git
git push -u origin main
```

**IEEE Access est en simple aveugle** : poussez sous votre propre compte et
mettez l'URL réelle directement dans le manuscrit (Data Availability) et dans
`CITATION.cff` (`repository-code`).

## 6. Zenodo

1. Connecter le compte GitHub sur zenodo.org, activer le dépôt.
2. Créer une release GitHub `v1.0.0` — Zenodo l'archive et émet un DOI.
3. Ajouter le DOI dans `CITATION.cff` et dans la section Data Availability.

Le DOI Zenodo est permanent là où une URL GitHub peut disparaître ; ajoutez-le
aussi dans la section Data Availability du manuscrit.

## 7. Après acceptation

Ajouter la référence du papier (DOI IEEE) dans `CITATION.cff` sous
`preferred-citation`.
