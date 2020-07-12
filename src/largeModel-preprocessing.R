checkConverged <- function(data)
{  
	print("alo pre split")
  dframes <- split(data, data$elementID*4+data$GaussPointID , drop=True)
  print("alo posle split")
  for(df in dframes)
  {
print("alo po frejmovima")	  
    df$converged <-  !rev(duplicated(rev(df$step)))
  }
  print("alo pre bind")
  data <- rbind(dframes)
  print("alo posle bind")
  return(data)
}


columnsToSelect <- c(6,7,8,9,10,12,11,13,14,4)
dt <- 0.001

# merge data
data <- read.csv("../data/tests/testLargeModel.csv")
data <- data[order(data$elementID, data$GaussPointID, data$step, data$iter),]
data <- checkConverged(data)
time <- data$step * dt
dataFinal <- data[,columnsToSelect]
dataFinal <- cbind(time, dataFinal)
dataFinal$testid <- data$elementID*4 + data$GaussPointID 
dataFull <- dataFinal


dataFull <- na.omit(dataFull)
write.csv(dataFull, file="../data/dataMexieLargeModel.csv",row.names=F, quote=F)
write.csv(dataFull[dataFull$converged,], file="../data/dataMexieLargeModelNoIter.csv",row.names=F, quote=F)

# iterations <- data %>% count(step)
# write.csv(iterations, file="iterations.csv",row.names=F, quote=F)
