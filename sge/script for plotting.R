library(readr)
library(dplyr)
library(data.table)
library(ggplot2)
library(patchwork)
library(ggpubr)
wd = "C:/Users/p288427/Desktop/megalomania/psge/"

#read and save the data
read_and_save = function(dir){
  setwd(dir)
  dirs = list.dirs()
  paths =  dirs[grepl("*run_\\d+", dirs)][-1]
  data = data.table()
  
  for (path in paths) {
    
    cat(path)
    
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
    print(" ->done")
  }
  colnames(data)[1:9]=c("gen","best_fit","genot","phenot","mut_prob","gram","fit_mean","fit_sd","best_error_test")
  
  write.csv(data, file = "data.csv")
  return(data)
}

#plot fitness over time of best individual across approaches and runs
fitness_over_time_across_approachs <- function(data){
  
  p <- ggplot(data %>%  filter(delay == "False"),
              aes(x = gen, 
                  y = best_fit,
                  color = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                  fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                  alpha = 0.0001)
  )+
    geom_line(linewidth = 0.1,
              linetype = "dotted",
              aes(group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay, run))
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
    facet_grid(cols = vars(start_mut_prob, prob_mut_probs, gauss, remap, delay)) +
    theme(legend.position = "None")
 
  layout = 'AAAA
            AAAA
            BBBB'

  x = q / z #+ plot_annotation(title = "Average performance of solutions over evolution")
  print(x + plot_layout(guides = 'collect', design = layout))
  ggsave("fitness_over_time_per_run_across_approachs_details.pdf",
         width = 14,
         height = 10,
         units = "cm")
  return(x)
}

### grouping is done by "approach"
add_violin_box_plot<- function(plot){
  return(
    plot +
      geom_violin( alpha = 0.1) + 
      geom_boxplot(alpha = 0.3, width = 0.1) +
      # geom_point() +
      geom_beeswarm(aes(color = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)), size = 0.5) +
      geom_pwc(
        group.by = "x.var",
        method = "t_test",
        label = "{p.adj}{p.adj.signif}",
        p.adjust.method = "BH",
        p.adjust.by = "group",
        hide.ns = F
      ) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.08))) + 
      scale_fill_viridis(discrete = T) +
      scale_color_viridis(discrete = T) +
      theme_bw() 
  )
}

###wrangle avg fitness data
wrangle_avg_final_fitness = function(data){
  return(data %>% 
           group_by(start_mut_prob, prob_mut_probs, gauss, remap, delay, run) %>% 
           filter(gen == max(gen)) %>% 
           summarise(avg_final_fitness = mean(-1 * best_fit)) %>% 
           ungroup()
  )
}
###wrangle avg fitness data
wrangle_avg_mid_fitness = function(data){
  return(data %>% 
           group_by(start_mut_prob, prob_mut_probs, gauss, remap, delay, run) %>% 
           filter(gen == max(gen)/4) %>% 
           summarise(avg_final_fitness = mean(-1 * best_fit)) %>% 
           ungroup()
  )
}

###Plot final best fitness 
make_best_fitness_final_t_test_boxplot = function(data){
  
  b = wrangle_avg_final_fitness(data)
  p <- b %>% ggplot(aes(x = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                        y = avg_final_fitness,
                        fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)
  )
  ) + 
    xlab("parameters combination")+
    ylab(paste("average best fitness in last generation", sep = ""))
  
  p = add_violin_box_plot(p)
  print(p)
  ggsave(paste("average_fitness_in_last_generation_across_approachs.jpg", sep = ""))
  ggsave(paste("average_fitness_in_last_generation_across_approachs.pdf", sep = ""),
         width = 14,
         height = 7,
         units ="cm")
  
  return(p)
}

###Plot final best fitness 
make_best_fitness_mid_t_test_boxplot = function(data){
  
  b = wrangle_avg_mid_fitness(data)
  p <- b %>% ggplot(aes(x = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                        y = avg_final_fitness,
                        fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)
  )
  ) + 
    xlab("parameters combination")+
    ylab(paste("average best fitness in last generation", sep = ""))
  
  p = add_violin_box_plot(p)
  print(p)
  
  return(p)
}


data = read_and_save(wd)

data = read.csv("data.csv")
data = data[-1]

fitness_over_time_across_approachs(data = data)
make_best_fitness_mid_t_test_boxplot(data)
fitness_over_time_across_approachs(data)

