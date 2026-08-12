# Avant de déposer — liste de contrôle

## 1. Copier vos résultats depuis `revision_outputs_v10/`

```
results/PAPER_RESULTS.md
results/all_results.json
results/scores/*.npz            <- 13 fichiers : T×5, K×5, PS×3
results/tables/*.csv
figures/fig_coverage_gap.png
figures/fig_coverage_price.png
figures/fig_ncal_price.png
figures/fig_mechanism.png
```

Vérifiez la taille totale des `.npz`. PaySim en compte deux vecteurs de
954 393 flottants par graine : comptez quelques dizaines de Mo. Au-delà de
100 Mo par fichier, GitHub refuse — dans ce cas, ne déposez ces fichiers que
sur Zenodo et laissez le lien dans `results/README.md`.

## 2. Compléter les champs marqués

| Fichier | À remplir |
|---|---|
| `CITATION.cff` | `repository-code`, et les ORCID si vous en avez |
| `data/README.md` | les quatre SHA-256 |
| le manuscrit | `[ANONYMISED REPOSITORY URL]` dans Data Availability |

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

**Pour la relecture en aveugle**, ne poussez pas sous votre nom. Deux options :
un dépôt sous compte neutre, ou un service d'anonymisation
(anonymous.4open.science). C'est cette URL qui va dans le manuscrit.

## 6. Zenodo

1. Connecter le compte GitHub sur zenodo.org, activer le dépôt.
2. Créer une release GitHub `v1.0.0` — Zenodo l'archive et émet un DOI.
3. Ajouter le DOI dans `CITATION.cff` et dans la section Data Availability.

Le DOI Zenodo est ce qu'un éditeur Elsevier attend : permanent, là où une URL
GitHub peut disparaître.

## 7. Après acceptation

Remplacer l'URL anonyme par l'URL réelle dans le manuscrit, et ajouter la
référence du papier dans `CITATION.cff` sous `preferred-citation`.
