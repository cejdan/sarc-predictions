#!/bin/bash -l


sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J RF_lasso --output "RF_lasso.txt" --wrap "Rscript 2021-04-06_RF_lasso_my0.9.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J SVM_lasso --output "SVM_lasso.txt" --wrap "Rscript 2021-04-06_SVM_lasso_my0.9.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J nnet_lasso --output "nnet_lasso.txt" --wrap "Rscript 2021-04-06_nnet_lasso_my0.9.R"

