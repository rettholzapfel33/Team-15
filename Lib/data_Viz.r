library(tidyverse)



x <- data.frame(group, count, task, treatment, time, age, gender)

ggplot(data = x) +
    geom_smooth(mapping = aes(x = count, y=group), color = "green") + 
    geom_smooth(mapping = aes(x = count, y=task), color = "blue") +
    geom_smooth(mapping = aes(x = count, y=treatment), color = "red") +
    geom_smooth(mapping = aes(x = count, y=time), color = "orange") +
    geom_smooth(mapping = aes(x = count, y=age), color = "brown") +
    geom_smooth(mapping = aes(x = count, y=gender), color = "purple") 


