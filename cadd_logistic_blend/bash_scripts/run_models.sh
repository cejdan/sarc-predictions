#!/bin/bash -l


sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 10_nnet --output "top10_nnet.txt" --wrap "Rscript 2021-03-21_CADD_pval_top10_0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 100_nnet --output "top100_nnet.txt" --wrap "Rscript 2021-03-21_CADD_pval_top100_0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 500_nnet --output "top500_nnet.txt" --wrap "Rscript 2021-03-21_CADD_pval_top500_0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 1000_nnet --output "top1000_nnet.txt" --wrap "Rscript 2021-03-21_CADD_pval_top1000_0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_nnet --output "top2000_nnet.txt" --wrap "Rscript 2021-03-21_CADD_pval_top2000_0.9_score_nnet.R"



sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 10_svm --output "top10_svm.txt" --wrap "Rscript 2021-03-21_CADD_pval_top10_0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 100_svm --output "top100_svm.txt" --wrap "Rscript 2021-03-21_CADD_pval_top100_0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 500_svm --output "top500_svm.txt" --wrap "Rscript 2021-03-21_CADD_pval_top500_0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 1000_svm --output "top1000_svm.txt" --wrap "Rscript 2021-03-21_CADD_pval_top1000_0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_svm --output "top2000_svm.txt" --wrap "Rscript 2021-03-21_CADD_pval_top2000_0.9_score_SVM.R"



sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_RF --output "top2000_RF.txt" --wrap "Rscript 2021-03-21_CADD_pval_top2000_0.9_score.R"

