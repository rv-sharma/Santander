options(shiny.maxRequestSize=500*1024^2)
options(shiny.port = 5000)
####################################
# Data Professor                   #
# http://youtube.com/dataprofessor #
# http://github.com/dataprofessor  #
####################################

# Modified from Winston Chang, 
# https://shiny.rstudio.com/gallery/shiny-theme-selector.html

# Concepts about Reactive programming used by Shiny, 
# https://shiny.rstudio.com/articles/reactivity-overview.html

# Load R packages

library(shiny)
library(shinythemes)
library(vroom)
library("e1071")
setwd("C:/Users/Admin/Documents/R/R Scripts/Shiny App")
model=readRDS('final_model.rds')
column_toDrop=readRDS('column_toDrop_list.rds')
# Define UI
ui <- fluidPage(theme = shinytheme("yeti"),
                navbarPage(
                   theme = "cerulean",  # <--- To use a theme, uncomment this
                   tags$h3('Santander Customer Transaction Prediction '),
  tabPanel( 
            
           tags$h4('App'),
           tags$h5('This is an app to predict whether a customer will make any transaction in the future or not, on the basis of past data.'),
  )),              
  sidebarPanel(fileInput("file", 'Upload CSV File', accept = c(".csv"))),
  mainPanel(textOutput(outputId = "Prediction"),
            downloadButton("downloadData", "Download"),
            tableOutput("head"))
  
) # fluidPage


# Define server function  
server <- function(input, output, session) {
  
  
  df=data()
  data <- reactive({
    req(input$file)
    
    df <- read.csv(input$file$datapath)
    
    df= df[, !colnames(df) %in% column_toDrop]
    
    x=subset(df,select=-ID_code)
    
    for (i in colnames(x)){
      x[,i] = (x[,i] - mean(x[,i])) / sd(x[,i])
    }
    
    df$target = predict(model, x, type = 'class')
    
    x =subset(df,select=c(ID_code,target))
    
    x
    
  })
  
  output$head <- renderTable(data())
  
  output$Prediction <- renderText('Predictions: ') 
  
  output$downloadData <- downloadHandler(
    filename = function() {
      paste("data-", "predictions", ".csv", sep="")
    },
    content = function(file) {
      write.csv(data(), file,row.names = FALSE)
    }
  )
  
}

# Create Shiny object
shinyApp(ui = ui, server = server)
