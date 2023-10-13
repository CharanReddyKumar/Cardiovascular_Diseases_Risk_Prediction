library(plotly)
library(corrplot)

ds <- read.csv("CVD_cleaned.csv")
convert_age_category <- function(category) {
  if (grepl("-", category)) {
    range <- as.numeric(strsplit(category, "-")[[1]])
    avg <- mean(range)
  } else {
    avg <- as.numeric(gsub("[^0-9]", "", category))
  }
  return(avg)
}

ds$Age_Category <- sapply(ds$Age_Category, convert_age_category)

plot_3d <- plot_ly(ds, x = ~BMI, y = ~Age_Category, z = ~Heart_Disease, 
                   type = "scatter3d", mode = "markers",
                   marker = list(size = 5, color = 'green', opacity = 0.7))

plot_3d <- plot_3d %>% layout(scene = list(
  xaxis = list(title = 'BMI', backgroundcolor = "rgb(200, 200, 230)"),
  yaxis = list(title = 'Age_Category', backgroundcolor = "rgb(230, 200, 230)"),
  zaxis = list(title = 'Heart_Disease', backgroundcolor = "rgb(200, 230, 200)")
),
margin = list(l = 0, r = 0, b = 0, t = 0))

plot_3d


