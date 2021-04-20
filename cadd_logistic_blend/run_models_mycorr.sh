#!/bin/bash -l


sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 10_nnet --output "top10_mycorr_nnet.txt" --wrap "Rscript 2021-04-02_CADD_pval_top10_my0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 100_nnet --output "top100_mycorr_nnet.txt" --wrap "Rscript 2021-04-02_CADD_pval_top100_my0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 500_nnet --output "top500_mycorr_nnet.txt" --wrap "Rscript 2021-04-02_CADD_pval_top500_my0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 1000_nnet --output "top1000_mycorr_nnet.txt" --wrap "Rscript 2021-04-02_CADD_pval_top1000_my0.9_score_nnet.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_nnet --output "top2000_mycorr_nnet.txt" --wrap "Rscript 2021-04-02_CADD_pval_top2000_my0.9_score_nnet.R"



sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 10_svm --output "top10_mycorr_svm.txt" --wrap "Rscript 2021-04-02_CADD_pval_top10_my0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 100_svm --output "top100_mycorr_svm.txt" --wrap "Rscript 2021-04-02_CADD_pval_top100_my0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 500_svm --output "top500_mycorr_svm.txt" --wrap "Rscript 2021-04-02_CADD_pval_top500_my0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 1000_svm --output "top1000_mycorr_svm.txt" --wrap "Rscript 2021-04-02_CADD_pval_top1000_my0.9_score_SVM.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_svm --output "top2000_mycorr_svm.txt" --wrap "Rscript 2021-04-02_CADD_pval_top2000_my0.9_score_SVM.R"



sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 10_RF --output "top10_mycorr_RF.txt" --wrap "Rscript 2021-04-02_CADD_pval_top10_my0.9_score_RF.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 100_RF --output "top100_mycorr_RF.txt" --wrap "Rscript 2021-04-02_CADD_pval_top100_my0.9_score_RF.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 500_RF --output "top500_mycorr_RF.txt" --wrap "Rscript 2021-04-02_CADD_pval_top500_my0.9_score_RF.R"

sbatch -A montgomery --partition serial --mem 5000 --time 3-0:00:00 -J 1000_RF --output "top1000_mycorr_RF.txt" --wrap "Rscript 2021-04-02_CADD_pval_top1000_my0.9_score_RF.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_RF --output "top2000_mycorr_RF.txt" --wrap "Rscript 2021-04-02_CADD_pval_top2000_my0.9_score_RF.R"
