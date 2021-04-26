#!/bin/bash -l


sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J NNET_LASSO --output "NNET_lasso_0.75.txt" --wrap "Rscript 2021-04-13_nnet_lasso_my0.75.R"
sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J NNET_LASSO --output "NNET_lasso_0.80.txt" --wrap "Rscript 2021-04-13_nnet_lasso_my0.80.R"
sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J NNET_LASSO --output "NNET_lasso_0.85.txt" --wrap "Rscript 2021-04-13_nnet_lasso_my0.85.R"
sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J NNET_LASSO --output "NNET_lasso_0.90.txt" --wrap "Rscript 2021-04-13_nnet_lasso_my0.90.R"
sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J NNET_LASSO --output "NNET_lasso_0.95.txt" --wrap "Rscript 2021-04-13_nnet_lasso_my0.95.R"