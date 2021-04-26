#!/bin/bash -l


sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_NNET --output "/s/cgm/Cejda/_thesis/fathmm_pval/NNET_top2000.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_NNET_top2000_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 1000_NNET --output "/s/cgm/Cejda/_thesis/fathmm_pval/NNET_top1000.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_NNET_top1000_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 500_NNET --output "/s/cgm/Cejda/_thesis/fathmm_pval/NNET_top500.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_NNET_top500_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 200_NNET --output "/s/cgm/Cejda/_thesis/fathmm_pval/NNET_top200.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_NNET_top200_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 100_NNET --output "/s/cgm/Cejda/_thesis/fathmm_pval/NNET_top100.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_NNET_top100_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 10_NNET --output "/s/cgm/Cejda/_thesis/fathmm_pval/NNET_top10.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_NNET_top10_my0.9_score.R"


sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_RF --output "/s/cgm/Cejda/_thesis/fathmm_pval/RF_top2000.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_RF_top2000_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 1000_RF --output "/s/cgm/Cejda/_thesis/fathmm_pval/RF_top1000.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_RF_top1000_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 500_RF --output "/s/cgm/Cejda/_thesis/fathmm_pval/RF_top500.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_RF_top500_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 200_RF --output "/s/cgm/Cejda/_thesis/fathmm_pval/RF_top200.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_RF_top200_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 100_RF --output "/s/cgm/Cejda/_thesis/fathmm_pval/RF_top100.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_RF_top100_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 10_RF --output "/s/cgm/Cejda/_thesis/fathmm_pval/RF_top10.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_RF_top10_my0.9_score.R"



sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 2000_SVM --output "/s/cgm/Cejda/_thesis/fathmm_pval/SVM_top2000.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_SVM_top2000_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 1000_SVM --output "/s/cgm/Cejda/_thesis/fathmm_pval/SVM_top1000.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_SVM_top1000_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 500_SVM --output "/s/cgm/Cejda/_thesis/fathmm_pval/SVM_top500.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_SVM_top500_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 200_SVM --output "/s/cgm/Cejda/_thesis/fathmm_pval/SVM_top200.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_SVM_top200_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 100_SVM --output "/s/cgm/Cejda/_thesis/fathmm_pval/SVM_top100.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_SVM_top100_my0.9_score.R"

sbatch -A montgomery --partition serial --mem 10000 --time 3-0:00:00 -J 10_SVM --output "/s/cgm/Cejda/_thesis/fathmm_pval/SVM_top10.txt" --wrap "Rscript 2021-04-14_FATHMM_pval_SVM_top10_my0.9_score.R"
