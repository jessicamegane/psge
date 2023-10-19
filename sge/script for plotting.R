
library(readr)
library(dplyr)
library(data.table)
wd = "C:/Users/p288427/Desktop/psge/"
setwd(wd)
dirs = list.dirs()
paths =  dirs[grepl("*run_\\d+", dirs)][-1]
data = data.table()

for (path in paths) {
  filename = file.path(path,"progress_report.csv")
  
  # Read the file contents as a character vector
  file_contents <- read_file(filename)
  # Substitute corrupted newlines followed by a space with just a newline
  cleaned_contents <- gsub("[\r\n]\\s", "\\s", file_contents)
  tempfile <- tempfile()
  writeLines(cleaned_contents, tempfile)
  
  # Read the CSV file from the temporary file
  tmp_data =  read.csv(tempfile, 
                   header = F,
                   sep = ";")
  
  properties <- read.table(text = path, sep = "/")[-1]
  colnames(properties) <- sub('_[^_]*$', '', properties)
  properties[1, ] <- sub('.*\\_', '', properties)
  
  data = rbind(data,cbind(tmp_data,properties))
  print(path)
}
saveRDS(data, file = "data.rds")
write.csv(data, file = "data.csv")
