library(readr)
library(plyr)
library(dplyr)
library(ggbeeswarm)
library(ggplot2)
library(ggpubr)
library(patchwork)
library(viridis)
wd = "C:/Users/p288427/Desktop/megalomania/"
setwd(wd)


#read and save the data
read_and_save = function(dir){
  setwd(dir)
  dirs = list.dirs()
  paths =  dirs[grepl("*run_\\d+", dirs)][-1]
  data = data.frame()
  
  for (path in paths) {
    cat(path)
    filename = file.path(path,"progress_report.csv")
    
    properties <- read.table(text = path, sep = "/")[-1]
    colnames(properties) <- c("alg_type",sub('_[^_]*$', '', properties[-1]))
    properties[1, -1] <- sub('.*\\_', '', properties[-1])
    if(!"remap" %in% colnames(properties)){
      properties = properties %>% mutate(remap = "False")
    } 
    
    tmp_data = data.frame(matrix(ncol = 9, nrow = 0))
    
    if (properties$alg_type == "psge") {
      
      
      # Read the file contents as a character vector
      file_contents <- read_file(filename)
      # Substitute corrupted newlines followed by a space with just a newline
      cleaned_contents <- gsub("[\r\n]\\s", "\\s", file_contents)
      tempfile <- tempfile()
      writeLines(cleaned_contents, tempfile)
      # Read the CSV file from the temporary file
      tmp_data = read.csv(tempfile, 
                          header = F,
                          sep = ";")
      colnames(tmp_data)[1:9]=c("gen","best_fit","genot","phenot","mut_prob","gram","fit_mean","fit_sd","best_error_test")
      
      
      
    } else  {
      colnames(tmp_data)[1:9]=c("gen","best_fit","genot","phenot","mut_prob","gram","fit_mean","fit_sd","best_error_test")
      csv = read.csv2(filename, 
                     header = F,
                     col.names = c("gen","best_fit","fit_mean","fit_sd"),
                     sep = "\t")
      tmp_data = rbind.fill(tmp_data, csv)
      
    }
    
    data = rbind(data,cbind(tmp_data,properties))
    print(" ->done")
  }
  
  write.csv(data, file = "data.csv")
  return(data)
}

#plot fitness over time of best individual across approaches and runs
fitness_over_time_across_approachs <- function(data){
  
  p <- ggplot(data %>%  filter(delay == "False"),
              aes(x = gen, 
                  y = best_fit,
                  color = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                  fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)
                  )
  ) +
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    # ylim(c(0, 1))+
    xlab("generations")+
    ylab(paste("average fitness", sep = "")) +
    theme_bw()
  
  q = p + geom_smooth(
    aes(group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay))
  )
  
  z = p + 
    geom_line(linewidth = 0.1,
              linetype = "dotted",
              aes(group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay, run))
    ) + 
    + geom_smooth(
      aes(group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay))
    )
    facet_grid(cols = rev(vars(start_mut_prob, prob_mut_probs, gauss, remap, delay))) +
    theme(legend.position = "None")
  
  layout = 'AAAA
            AAAA
            BBBB'
  
  x = q / z + plot_annotation(title = "Average performance of solutions over evolution")
  print(x + plot_layout(guides = 'collect', design = layout))
  ggsave("fitness_over_time_per_run_across_approachs_details.pdf",
         width = 14,
         height = 10,
         units = "cm")
  return(x)
}

### add violin plots with nice aestethics
add_violin_box_plot<- function(plot){
  return(
    plot +
      geom_violin( alpha = 0.1) + 
      geom_boxplot(alpha = 0.3, width = 0.1) +
      geom_beeswarm(size = 0.5) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.08))) + 
      scale_fill_viridis(discrete = T) +
      scale_color_viridis(discrete = T) +
      theme_bw() 
  )
}

### add t_test comparison bars
add_t_test = function(plot){
  return(
    plot +
      geom_pwc(
        group.by = "x.var",
        method = "t_test",
        label = "{p.adj}{p.adj.signif}",
        p.adjust.method = "BH",
        p.adjust.by = "group",
        hide.ns = T
      )
  )
}

###wrangle avg fitness data of gen n
wrangle_avg_fitness_gen = function(data, n_gen){
  return(data %>% 
           group_by(start_mut_prob, prob_mut_probs, gauss, remap, delay, run) %>% 
           filter(gen == n_gen) %>% 
           dplyr::summarise(avg_fitness = mean(-1 * best_fit)) %>% 
           ungroup()
  )
}

###Plot best fitness boxplot of generation n
make_best_fitness_t_test_boxplot_gen = function(data, gen){

  b = wrangle_avg_fitness_gen(data, gen)
  p <- b %>% ggplot(aes(x = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                        y = avg_fitness,
                        fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                        color = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)
  )
  ) + 
    xlab("parameters combination")+
    ylab(paste("average best fitness in generation ",gen, sep = ""))
  
  p = add_violin_box_plot(p)
  print(p)
  ggsave(paste("average_fitness_in_",gen,"_generation_across_approachs.jpg", sep = ""))
  ggsave(paste("average_fitness_in_",gen,"_generation_across_approachs.pdf", sep = ""),
         width = 14,
         height = 7,
         units ="cm")
  
  p_t_test = add_t_test(p)
  print(p_t_test)
  ggsave(paste("t_test_average_fitness_in_",gen,"_generation_across_approachs.jpg", sep = ""))
  
  return(p)
}

#use when need loading from raw data and not .csv
#
data = read_and_save(wd)
data = read.csv2("data.csv", header = T, sep =",")

fitness_over_time_across_approachs(data = data %>% subset(alg_type == "co-psge"))
fitness_over_time_across_approachs(data = data %>% subset(alg_type == "psge"))
make_best_fitness_t_test_boxplot_gen(data %>% subset(alg_type == "co-psge"), gen = 250)
make_best_fitness_t_test_boxplot_gen(data = data %>% subset(alg_type == "co-psge"), gen = 500)
make_best_fitness_t_test_boxplot_gen(data = data %>% subset(alg_type == "co-psge"), gen = 1000)
make_best_fitness_t_test_boxplot_gen(data = data %>% subset(alg_type == "psge"), gen = 1000)
