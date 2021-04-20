

library(ggplot2)
library(caret)
library(pROC)

results_df = data.frame(model = NA, feature_selection = NA, sample = NA, rsq_threshold = NA, 
                        AUC = NA, sigma = NA, C = NA, accuracy = NA, kappa = NA, 
                        accuracy_null = NA, accuracy_pval = NA, sensitivity = NA, specificity = NA
)

{
  # Neural Network - Logistic - top10 - 0.95
  #model = "NNET"
  #feature_selection = "logistic"
  #sample = "top10"
  #rsq_threshold = 0.95
  
  #setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top10")
  #rds = readRDS("tuned_SVM_top10_mycorr0.95.rds")
  #test = read.csv("simplified_top10_0.95_test.csv")
  
  #x_test = model.matrix(PHENOTYPE ~ ., test)[,c(-1)]
  #y_test = as.factor(test[,2])
  #levels(y_test) = c("control", "sarc")
  #pred_probs = predict(rds, x_test, type = "prob")
  #pred_class = predict(rds, x_test)
  #cmatrix = confusionMatrix(pred_class, y_test, positive = "sarc")
  #accuracy = as.numeric(cmatrix$overall[1])
  #kappa = as.numeric(cmatrix$overall[2])
  #accuracy_null = as.numeric(cmatrix$overall[5])
  #accuracy_pval = as.numeric(cmatrix$overall[6])
  #sensitivity = as.numeric(cmatrix$byClass[1])
  #specificity = as.numeric(cmatrix$byClass[2])
  
  #ROC = roc(y_test, pred_probs[,2])
  #AUC = auc(ROC)
  #mtry = rds$bestTune[,1]
  #temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
  #                     AUC = AUC, mtry = mtry, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
  #                     accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  #results_df = rbind(results_df, temp_df)
  #results_df = results_df[2,]
  #row.names(results_df) = NULL
  
  
  ############
  # Top 100
  
  
  # Neural Network - Logistic - top100 - 0.95
  model = "SVM"
  feature_selection = "logistic"
  sample = "top100"
  rsq_threshold = 0.95
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
  rds = readRDS("tuned_SVM_top100_mycorr0.95.rds")
  test = read.csv("simplified_top100_0.95_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  results_df = results_df[2,]
  row.names(results_df) = NULL
  
  
  # Neural Network - Logistic - top100 - 0.90
  model = "NNET"
  feature_selection = "logistic"
  sample = "top100"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
  rds = readRDS("tuned_SVM_top100_mycorr0.90.rds")
  test = read.csv("simplified_top100_0.90_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top100 - 0.85
  model = "NNET"
  feature_selection = "logistic"
  sample = "top100"
  rsq_threshold = 0.85
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
  rds = readRDS("tuned_SVM_top100_mycorr0.85.rds")
  test = read.csv("simplified_top100_0.85_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top100 - 0.80
  model = "NNET"
  feature_selection = "logistic"
  sample = "top100"
  rsq_threshold = 0.80
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
  rds = readRDS("tuned_SVM_top100_mycorr0.80.rds")
  test = read.csv("simplified_top100_0.80_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top100 - 0.75
  model = "NNET"
  feature_selection = "logistic"
  sample = "top100"
  rsq_threshold = 0.75
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top100")
  rds = readRDS("tuned_SVM_top100_mycorr0.75.rds")
  test = read.csv("simplified_top100_0.75_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  ###########################
  # Top 500
  
  
  # Neural Network - Logistic - top500 - 0.95
  model = "NNET"
  feature_selection = "logistic"
  sample = "top500"
  rsq_threshold = 0.95
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
  rds = readRDS("tuned_SVM_top500_mycorr0.95.rds")
  test = read.csv("simplified_top500_0.95_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top500 - 0.90
  model = "NNET"
  feature_selection = "logistic"
  sample = "top500"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
  rds = readRDS("tuned_SVM_top500_mycorr0.90.rds")
  test = read.csv("simplified_top500_0.90_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top500 - 0.85
  model = "NNET"
  feature_selection = "logistic"
  sample = "top500"
  rsq_threshold = 0.85
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
  rds = readRDS("tuned_SVM_top500_mycorr0.85.rds")
  test = read.csv("simplified_top500_0.85_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top500 - 0.80
  model = "NNET"
  feature_selection = "logistic"
  sample = "top500"
  rsq_threshold = 0.80
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
  rds = readRDS("tuned_SVM_top500_mycorr0.80.rds")
  test = read.csv("simplified_top500_0.80_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  # Neural Network - Logistic - top500 - 0.75
  model = "NNET"
  feature_selection = "logistic"
  sample = "top500"
  rsq_threshold = 0.75
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top500")
  rds = readRDS("tuned_SVM_top500_mycorr0.75.rds")
  test = read.csv("simplified_top500_0.75_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top1000 - 0.95
  model = "NNET"
  feature_selection = "logistic"
  sample = "top1000"
  rsq_threshold = 0.95
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
  rds = readRDS("tuned_SVM_top1000_mycorr0.95.rds")
  test = read.csv("simplified_top1000_0.95_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top1000 - 0.90
  model = "NNET"
  feature_selection = "logistic"
  sample = "top1000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
  rds = readRDS("tuned_SVM_top1000_mycorr0.90.rds")
  test = read.csv("simplified_top1000_0.90_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top1000 - 0.85
  model = "NNET"
  feature_selection = "logistic"
  sample = "top1000"
  rsq_threshold = 0.85
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
  rds = readRDS("tuned_SVM_top1000_mycorr0.85.rds")
  test = read.csv("simplified_top1000_0.85_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  # Neural Network - Logistic - top1000 - 0.80
  model = "NNET"
  feature_selection = "logistic"
  sample = "top1000"
  rsq_threshold = 0.80
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
  rds = readRDS("tuned_SVM_top1000_mycorr0.80.rds")
  test = read.csv("simplified_top1000_0.80_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top1000 - 0.75
  model = "NNET"
  feature_selection = "logistic"
  sample = "top1000"
  rsq_threshold = 0.75
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top1000")
  rds = readRDS("tuned_SVM_top1000_mycorr0.75.rds")
  test = read.csv("simplified_top1000_0.75_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - Logistic - top2000 - 0.95
  model = "NNET"
  feature_selection = "logistic"
  sample = "top2000"
  rsq_threshold = 0.95
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
  rds = readRDS("tuned_SVM_top2000_mycorr0.95.rds")
  test = read.csv("simplified_top2000_0.95_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top2000 - 0.90
  model = "NNET"
  feature_selection = "logistic"
  sample = "top2000"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
  rds = readRDS("tuned_SVM_top2000_mycorr0.90.rds")
  test = read.csv("simplified_top2000_0.90_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top2000 - 0.85
  model = "NNET"
  feature_selection = "logistic"
  sample = "top2000"
  rsq_threshold = 0.85
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
  rds = readRDS("tuned_SVM_top2000_mycorr0.85.rds")
  test = read.csv("simplified_top2000_0.85_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top2000 - 0.80
  model = "NNET"
  feature_selection = "logistic"
  sample = "top2000"
  rsq_threshold = 0.80
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
  rds = readRDS("tuned_SVM_top2000_mycorr0.80.rds")
  test = read.csv("simplified_top2000_0.80_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  # Neural Network - Logistic - top2000 - 0.75
  model = "NNET"
  feature_selection = "logistic"
  sample = "top2000"
  rsq_threshold = 0.75
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist1_logistic/top2000")
  rds = readRDS("tuned_SVM_top2000_mycorr0.75.rds")
  test = read.csv("simplified_top2000_0.75_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  # Neural Network - LASSO - 0.95
  model = "NNET"
  feature_selection = "LASSO"
  sample = "LASSO"
  rsq_threshold = 0.95
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
  rds = readRDS("tuned_svmRadial_lasso_mycorr0.95.rds")
  test = read.csv("simplified_lasso_0.95_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - LASSO - 0.90
  model = "NNET"
  feature_selection = "LASSO"
  sample = "LASSO"
  rsq_threshold = 0.90
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
  rds = readRDS("tuned_svmRadial_lasso_mycorr0.90.rds")
  test = read.csv("simplified_lasso_0.90_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - LASSO - 0.85
  model = "NNET"
  feature_selection = "LASSO"
  sample = "LASSO"
  rsq_threshold = 0.85
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
  rds = readRDS("tuned_svmRadial_lasso_mycorr0.85.rds")
  test = read.csv("simplified_lasso_0.85_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  # Neural Network - LASSO - 0.80
  model = "NNET"
  feature_selection = "LASSO"
  sample = "LASSO"
  rsq_threshold = 0.80
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
  rds = readRDS("tuned_svmRadial_lasso_mycorr0.80.rds")
  test = read.csv("simplified_lasso_0.80_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  # Neural Network - LASSO - 0.75
  model = "NNET"
  feature_selection = "LASSO"
  sample = "LASSO"
  rsq_threshold = 0.75
  
  setwd("//jetsam/scratch/cgm/Cejda/_thesis/snplist2_lasso")
  rds = readRDS("tuned_svmRadial_lasso_mycorr0.75.rds")
  test = read.csv("simplified_lasso_0.75_test.csv")
  
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
  sigma = rds$bestTune[,1]
  C = rds$bestTune[,2]
  temp_df = data.frame(model = model, feature_selection = feature_selection, sample = sample, rsq_threshold = rsq_threshold, 
                       AUC = AUC, sigma = sigma, C = C, accuracy = accuracy, kappa = kappa, accuracy_null = accuracy_null,
                       accuracy_pval = accuracy_pval, sensitivity = sensitivity, specificity = specificity)
  
  results_df = rbind(results_df, temp_df)
  
  
  
  
}



results_df$sample = factor(results_df$sample,levels = c("top100", "top500", "top1000", "top2000", "LASSO"))

ggplot(data = results_df, aes(x = sample, y=AUC, fill = rsq_threshold)) +
  geom_bar(stat="identity", position=position_dodge2(preserve = "single")) +
  coord_cartesian(ylim = c(0.5,0.61)) +
  ggtitle("SVM Radial Kernel - AUC")


ggplot(data = results_df, aes(x = sample, y=kappa, fill = rsq_threshold)) +
  geom_bar(stat="identity", position=position_dodge2(preserve = "single")) +
  coord_cartesian(ylim = c(0,0.2)) +
  ggtitle("SVM Radial Kernel - Kappa") +
  scale_fill_gradient(low = "green4", high = "lightgreen")


ease_of_use = results_df[,c(3,4,8,11,9,12,13,5)]
write.table(ease_of_use, "easy_svm_results.csv", sep=",", quote=F, row.names = F)
