getAccuracyAndGini2 <- function(data,
                                predicted_probs,
                                target_variable = "UCURNINS",
                                target_levels = c("Yes", "No"),
                                predicted_class = "Yes",
                                cutoff = 0.5) {
  
  # funkcja dla podanych prawdopodobieństw sukcesu obliczy 
  # accuracy, specificity, sensitivity i Gini = 2 * AUC - 1
  library(pROC)
  forecasts_p <- data[, predicted_probs]
  
  # wygeneruj przewidywaną kategorię
  forecasts_c <- 
    factor(ifelse(forecasts_p > cutoff,
                  predicted_class,
                  target_levels[which(target_levels != predicted_class)]),
           levels = target_levels)
  
  # wartości rzeczywiste
  real <- (data[, target_variable])
  
  # pole pod wykresem ROC
  AUC <- roc(predictor = forecasts_p, response = real, quiet = T)
  
  # tabela klasyfikacji i miary na niej oparte
  table <- confusionMatrix(forecasts_c,
                           real,
                           predicted_class) 
  
  # zbieramy w ostateczny wynik
  result <- c(table$overall[1], # Accuracy
              table$byClass[1:2], # sens, spec
              Gini = 2 * AUC$auc - 1)
  
  return(result)
}
