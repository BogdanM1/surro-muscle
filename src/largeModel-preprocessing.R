checkConverged <- function(data)
{  
  dframes <- split(data, data$elementID*4+data$GaussPointID , drop=True)
  rm(data)
  first <- TRUE
  for(df in dframes)
  {
    df$converged <-  !rev(duplicated(rev(df$step)))
    if(first) 
    {
      res_data <- df
      first <- FALSE
    } 
    else
    { 
      res_data <- rbind(res_data, df)
    }   
  }
  return(res_data)
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

dataFinal <- na.omit(dataFinal)
write.csv(dataFinal, file="../data/dataMexieLargeModel.csv",row.names=F, quote=F)
write.csv(dataFinal[dataFinal$converged,], file="../data/dataMexieLargeModelNoIter.csv",row.names=F, quote=F)

# iterations <- data %>% count(step)
# write.csv(iterations, file="iterations.csv",row.names=F, quote=F)
