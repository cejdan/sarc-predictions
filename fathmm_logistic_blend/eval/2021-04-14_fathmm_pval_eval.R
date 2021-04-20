

library(ggplot2)
library(caret)
library(pROC)

results_df = data.frame(model = NA, feature_selection = NA, sample = NA, rsq_threshold = NA, 
                        AUC = NA, mtry = NA, sigma = NA, c = NA, size = NA, decay = NA, accuracy = NA, kappa = NA, 
                        accuracy_null = NA, accuracy_pval = NA, sensitivity = NA, specificity = NA
)

{
  # Random Forest - FATHMM + Logistic - top10 - 0.9
  model = "RF"
  feature_selection = "FATHMM + Logistic"
  sample = "top10"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_forest_fathmm_pval_top10_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top10_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = rds$bestTune[,1]
  sigma = NA
  c = NA
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  results_df = results_df[2,]
  row.names(results_df) = NULL
  
  
  # Random Forest - FATHMM + Logistic - top100 - 0.9
  model = "RF"
  feature_selection = "FATHMM + Logistic"
  sample = "top100"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_forest_fathmm_pval_top100_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top100_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = rds$bestTune[,1]
  sigma = NA
  c = NA
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Random Forest - FATHMM + Logistic - top500 - 0.9
  model = "RF"
  feature_selection = "FATHMM + Logistic"
  sample = "top500"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_forest_fathmm_pval_top500_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top500_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = rds$bestTune[,1]
  sigma = NA
  c = NA
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  # Random Forest - FATHMM + Logistic - top1000 - 0.9
  model = "RF"
  feature_selection = "FATHMM + Logistic"
  sample = "top1000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_forest_fathmm_pval_top1000_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top1000_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = rds$bestTune[,1]
  sigma = NA
  c = NA
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  
  
  # Random Forest - FATHMM + Logistic - top2000 - 0.9
  model = "RF"
  feature_selection = "FATHMM + Logistic"
  sample = "top2000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_forest_fathmm_pval_top2000_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top2000_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = rds$bestTune[,1]
  sigma = NA
  c = NA
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # NNET - FATHMM + Logistic - top10 - 0.9
  model = "NNET"
  feature_selection = "FATHMM + Logistic"
  sample = "top10"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_nnet_fathmm_pval_top10_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top10_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = NA
  c = NA
  size = rds$bestTune[,1]
  decay = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # NNET - FATHMM + Logistic - top100 - 0.9
  model = "NNET"
  feature_selection = "FATHMM + Logistic"
  sample = "top100"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_nnet_fathmm_pval_top100_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top100_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = NA
  c = NA
  size = rds$bestTune[,1]
  decay = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # NNET - FATHMM + Logistic - top500 - 0.9
  model = "NNET"
  feature_selection = "FATHMM + Logistic"
  sample = "top500"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_nnet_fathmm_pval_top500_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top500_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = NA
  c = NA
  size = rds$bestTune[,1]
  decay = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # NNET - FATHMM + Logistic - top1000 - 0.9
  model = "NNET"
  feature_selection = "FATHMM + Logistic"
  sample = "top1000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_nnet_fathmm_pval_top1000_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top1000_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = NA
  c = NA
  size = rds$bestTune[,1]
  decay = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # NNET - FATHMM + Logistic - top2000 - 0.9
  model = "NNET"
  feature_selection = "FATHMM + Logistic"
  sample = "top2000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_nnet_fathmm_pval_top2000_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top2000_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "2")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = NA
  c = NA
  size = rds$bestTune[,1]
  decay = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # SVM - FATHMM + Logistic - top10 - 0.9
  model = "SVM"
  feature_selection = "FATHMM + Logistic"
  sample = "top10"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_svmRadial_fathmm_pval_top10_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top10_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  levels(y_test) = c("control", "sarc")
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = rds$bestTune[,1]
  c = rds$bestTune[,2]
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # SVM - FATHMM + Logistic - top100 - 0.9
  model = "SVM"
  feature_selection = "FATHMM + Logistic"
  sample = "top100"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_svmRadial_fathmm_pval_top100_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top100_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  levels(y_test) = c("control", "sarc")
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = rds$bestTune[,1]
  c = rds$bestTune[,2]
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # SVM - FATHMM + Logistic - top500 - 0.9
  model = "SVM"
  feature_selection = "FATHMM + Logistic"
  sample = "top500"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_svmRadial_fathmm_pval_top500_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top500_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  levels(y_test) = c("control", "sarc")
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = rds$bestTune[,1]
  c = rds$bestTune[,2]
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # SVM - FATHMM + Logistic - top1000 - 0.9
  model = "SVM"
  feature_selection = "FATHMM + Logistic"
  sample = "top1000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_svmRadial_fathmm_pval_top1000_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top1000_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  levels(y_test) = c("control", "sarc")
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = rds$bestTune[,1]
  c = rds$bestTune[,2]
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # SVM - FATHMM + Logistic - top2000 - 0.9
  model = "SVM"
  feature_selection = "FATHMM + Logistic"
  sample = "top2000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
  rds = readRDS("tuned_svmRadial_fathmm_pval_top2000_mycorr0.9_score.rds")
  test = read.csv("simplified_fathmm_pval_top2000_my0.9_SCORES_test.csv")
  
  x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  y_test = as.factor(test[,2])
  levels(y_test) = c("control", "sarc")
  pred_probs = predict(rds, x_test, type = "prob")
  pred_class = predict(rds, x_test)
  cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
  accuracy = as.numeric(cmatrix$overall[1])
  kappa = as.numeric(cmatrix$overall[2])
  accuracy_null = as.numeric(cmatrix$overall[5])
  accuracy_pval = as.numeric(cmatrix$overall[6])
  sensitivity = as.numeric(cmatrix$byClass[1])
  specificity = as.numeric(cmatrix$byClass[2])
  
  ROC = roc(y_test, pred_probs[,2])
  AUC = auc(ROC)
  mtry = NA
  sigma = rds$bestTune[,1]
  c = rds$bestTune[,2]
  size = NA
  decay = NA
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, mtry = mtry, sigma = sigma, c = c, size = size, decay = decay, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
}







# Plot results 

results_df$sample = factor(results_df$sample,levels = c("top10", "top100", "top500", "top1000", "top2000"))

ggplot(data = results_df, aes(x = sample, y=AUC, fill = model)) +
  geom_bar(stat="identity", position=position_dodge2(preserve = "single")) +
  coord_cartesian(ylim = c(0.45,0.61)) +
  ggtitle("FATHMM + Logistic blended models - AUC") +
  scale_fill_viridis_d(begin = 0, end = 0.5, option = "magma") +
  theme(axis.text.y = element_text(size=10, face = "bold"), axis.text.x = element_text(size=10, face = "bold"))


ggplot(data = results_df, aes(x = sample, y=kappa, fill = model)) +
  geom_bar(stat="identity", position=position_dodge2(preserve = "single")) +
  coord_cartesian(ylim = c(0,0.17)) +
  ggtitle("FATHMM + Logistic blended models - Kappa") +
  scale_fill_viridis_d(begin = 0, end = 0.5, option = "viridis") +
  theme(axis.text.y = element_text(size=10, face = "bold"), axis.text.x = element_text(size=10, face = "bold"))




ease_of_use = results_df[,c(1,3,11,14,12,15,16,5)] # sample, 
write.table(ease_of_use, "easy_fathmm_logistic_results.csv", sep=",", quote=F, row.names = F)
