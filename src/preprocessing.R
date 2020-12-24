checkConverged <- function(data)
{
  data$converged <-  !rev(duplicated(rev(data$step)))
  return(data)
}

removeGaussPoint <- function(data)
{
  data <- data[seq(4,nrow(data),4),]
  return(data) 
}

columnsToSelect <- c(6,7,8,9,10,12,11,13,14,4)
dt <- 0.001

# merge data
data <- read.csv("../data/tests/test1.csv")
data <- removeGaussPoint(data)
data <- checkConverged(data)
time <- data$step * dt
data <- data[,columnsToSelect]
data <- cbind(time, data)
data$testid <- 1
dataFull <- data

for (i in c(2:85))
{ 
  data <- read.csv(sprintf("../data/tests/test%d.csv",i))
  data <- removeGaussPoint(data)
  data <- checkConverged(data)  
  time <- data$step * dt  
  data <- data[,columnsToSelect]  
  data <- cbind(time, data)
  data$testid <- i
  dataFull <- rbind(dataFull, data)
}

dataFull <- na.omit(dataFull)
write.csv(dataFull, file="../data/dataMexie.csv",row.names=F, quote=F)
write.csv(dataFull[dataFull$converged,], file="../data/dataMexieNoIter.csv",row.names=F, quote=F)

# iterations <- data %>% count(step)
# write.csv(iterations, file="iterations.csv",row.names=F, quote=F)
