removeIterations <- function(data)
{
  data <-  data[!rev(duplicated(rev(data$step))),]
  return(data)
}

removeGaussPoint <- function(data)
{
  data <- data[seq(4,nrow(data),4),]
  return(data) 
}

columnsToSelect <- c(6,7,8,9,11,10,12,4)
dt <- 0.001

# merge data
data <- read.csv("../data/tests/test1.csv")
data <- removeGaussPoint(data)
#data <- removeIterations(data)
time <- data$step * dt
data <- data[,columnsToSelect]
data <- cbind(time, data)
data$testid <- 1
dataFull <- data

for (i in c(2:8))
{ 
  data <- read.csv(sprintf("../data/tests/test%d.csv",i))
  data <- removeGaussPoint(data)
 # data <- removeIterations(data)  
  time <- data$step * dt  
  data <- data[,columnsToSelect]  
  data <- cbind(time, data)
  data$testid <- i
  dataFull <- rbind(dataFull, data)
}

write.csv(dataFull, file="../data/dataMexie.csv",row.names=F, quote=F)

# iterations <- data %>% count(step)
# write.csv(iterations, file="iterations.csv",row.names=F, quote=F)
