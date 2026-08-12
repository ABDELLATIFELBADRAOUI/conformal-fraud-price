# Résultats du papier simplifié

Graines : [42, 43, 44, 45, 46]  |  découpage 55%/15%/15%/reste

## A — Couverture marginale contre couverture conditionnelle

                   marg_cov_marginal  marg_cov_fraud  mond_cov_fraud  mond_alert_pct
dataset alpha_sig                                                                   
K       0.05                  0.9729          0.0087          0.9744         73.6469
        0.10                  0.9729          0.0087          0.8871         22.2350
        0.20                  0.8615          0.0087          0.7822          0.7557
PS      0.05                  0.9880          0.4516          0.9571          0.4144
        0.10                  0.9615          0.2250          0.9148          0.3934
        0.20                  0.9615          0.2250          0.8248          0.3515
T       0.05                  0.9778          0.1962          1.0000        100.0000
        0.10                  0.9245          0.1962          0.9154         13.3154
        0.20                  0.8016          0.1115          0.8039          0.5346

Proportion de graines où la couverture des fraudes atteint la cible :

                   marg_fraud_ok  mond_fraud_ok
dataset alpha_sig                              
K       0.05                 0.0           0.80
        0.10                 0.0           0.20
        0.20                 0.0           0.00
PS      0.05                 0.0           1.00
        0.10                 0.0           0.67
        0.20                 0.0           0.67
T       0.05                 0.0           1.00
        0.10                 0.0           0.80
        0.20                 0.0           0.80

## C — Le prix de la garantie

                               achieved_fraud_coverage  alert_volume_pct  lambda_approved_pct  base_rate_pct
dataset target_fraud_coverage                                                                               
K       0.50                                    0.5320            0.3572               0.1813         0.3860
        0.55                                    0.5897            0.4216               0.1590         0.3860
        0.60                                    0.6180            0.4491               0.1481         0.3860
        0.65                                    0.6807            0.5325               0.1239         0.3860
        0.70                                    0.6893            0.5441               0.1206         0.3860
        0.75                                    0.7214            0.5904               0.1082         0.3860
        0.80                                    0.7822            0.7557               0.0847         0.3860
        0.85                                    0.8265            0.9958               0.0676         0.3860
        0.90                                    0.8871           22.2350               0.0562         0.3860
        0.95                                    0.9744           73.6469               0.0378         0.3860
PS      0.50                                    0.6731            0.2829               0.1377         0.4200
        0.55                                    0.6731            0.2829               0.1377         0.4200
        0.60                                    0.6731            0.2829               0.1377         0.4200
        0.65                                    0.6788            0.2855               0.1353         0.4200
        0.70                                    0.7152            0.3015               0.1199         0.4200
        0.75                                    0.8050            0.3425               0.0822         0.4200
        0.80                                    0.8248            0.3515               0.0738         0.4200
        0.85                                    0.8542            0.3653               0.0614         0.4200
        0.90                                    0.9148            0.3934               0.0359         0.4200
        0.95                                    0.9571            0.4144               0.0181         0.4200
T       0.50                                    0.5346            0.0665               0.0567         0.1217
        0.55                                    0.5731            0.0735               0.0520         0.1217
        0.60                                    0.5769            0.0744               0.0515         0.1217
        0.65                                    0.6308            0.0847               0.0450         0.1217
        0.70                                    0.6923            0.0988               0.0375         0.1217
        0.75                                    0.7154            0.1058               0.0347         0.1217
        0.80                                    0.8039            0.5346               0.0240         0.1217
        0.85                                    0.8039            0.5346               0.0240         0.1217
        0.90                                    0.9154           13.3154               0.0118         0.1217
        0.95                                    1.0000          100.0000                  NaN         0.1217

## Statistique KS

         ks_marginal  ks_fraud
dataset                       
K             0.0090    0.0509
PS            0.0080    0.0253
T             0.0354    0.1148

## Modèle et comparateurs

                            auprc           raw_cost               ece        
                             mean     std       mean       std    mean     std
dataset model                                                                 
K       ADAPTIVE-CP-FRAUD  0.5288  0.0433  7382.5000  195.1360  0.0004  0.0003
        Baseline SPW       0.5097  0.0499  7625.3000  154.3798  0.0002  0.0001
        HybridMeta-XGB     0.5144  0.0381  8005.9000  630.5720  0.0003  0.0003
PS      ADAPTIVE-CP-FRAUD  0.9885  0.0006   502.3333  120.1013  0.0005  0.0000
        Baseline SPW       0.9942  0.0004   474.0000  242.1838  0.0004  0.0000
        HybridMeta-XGB     0.9941  0.0002   315.3333   43.4665  0.0004  0.0000
T       ADAPTIVE-CP-FRAUD  0.7125  0.0123   154.0000    3.6818  0.0001  0.0000
        Baseline SPW       0.7158  0.0212   155.2000   15.2956  0.0002  0.0001
        HybridMeta-XGB     0.7373  0.0120   138.8000    2.5298  0.0001  0.0000

## Ablation — coût brut

                               mean    std
dataset config                            
K       baseline_no_anomaly  7625.3  154.4
        full                 7382.5  195.1
        no_stage1_uniform    7444.0  539.9
        no_stage2_spw        8001.5  374.6
PS      baseline_no_anomaly   474.0  242.2
        full                  502.3  120.1
        no_stage1_uniform     402.7   37.3
        no_stage2_spw         755.3  430.9
T       baseline_no_anomaly   155.2   15.3
        full                  154.0    3.7
        no_stage1_uniform     156.2    7.9
        no_stage2_spw         137.3    3.0

Graines où la configuration bat le pipeline complet — K : {'baseline_no_anomaly': 2, 'no_stage1_uniform': 7, 'no_stage2_spw': 1} sur 10

Graines où la configuration bat le pipeline complet — PS : {'baseline_no_anomaly': 2, 'no_stage1_uniform': 2, 'no_stage2_spw': 1} sur 3

Graines où la configuration bat le pipeline complet — T : {'baseline_no_anomaly': 5, 'no_stage1_uniform': 5, 'no_stage2_spw': 10} sur 10

## Résultats négatifs

Sommet de $\alpha^*$ par jeu et par graine :

dataset vertex  n
      K     IF 10
     PS    LOF  3
      T     IF 10

SemiSync — itérations et nombre de valeurs distinctes de $\phi$ :

        n_iter     phi_unique    
          mean max       mean max
dataset                          
K          2.0   2        1.0   1
PS         2.0   2        1.0   1
T          2.0   2        1.0   1
