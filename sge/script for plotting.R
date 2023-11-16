library(data.table)
library(dplyr)
library(ggbeeswarm)
library(ggplot2)
library(ggpubr)
library(jsonlite)
library(plyr)
library(patchwork)
library(kableExtra)
library(readr)
library(rstatix)
library(tidyr)
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
    
    #extract the parameters of the run from its path
    properties <- read.table(text = path, sep = "/")[-1]
    colnames(properties) <- c("alg_type",sub('_[^_]*$', '', properties[-1]))
    properties[1, -1] <- sub('.*\\_', '', properties[-1])
    #if there is no remap folder then add column with remap = False
    if(!"remap" %in% colnames(properties)){
      properties = properties %>% mutate(remap = "False")
    } 
    
    tmp_data = data.frame(matrix(ncol = 9, nrow = 0))
    
    #if psge clean the corrupted newlines due to python saving matrices and reread as cleaned csv
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

#transofrms data of psge so that mutation porbabilities can be read
transform_psge_data_to_read_mut_probs = function(data){
  a = data %>% subset(alg_type == "psge") %>% 
    #Fixing fromatting of data saved from python for gram
    mutate(gram = gsub("(\\d+)\\.(,|\\])","\\1\\2", #remove all trailing . after numbers (which are always followed by either a , or a ])
                       gsub(", ]","]", #remove extra space before ] 
                            gsub("\\s+",", ",  #remove all multiple white spaces and add commas
                                 gsub("s"," ",gram)))) #remove weird s artifact that appears instead of spaces
    ) %>% 
    #Reading from JSON format to R matrix for grammar prob
    mutate(gram = map(gram, ~fromJSON(as.character(.x)))) %>% 
    #Reading from JSON format to R matrix for mutation prob
    mutate(mut_prob = map(mut_prob, ~fromJSON(as.character(.x)))) %>%
    #unrolling the mut_prob matrix column into separate columns in tidy format
    mutate(mut_prob = map(mut_prob, ~as.data.frame(.x) %>% rowid_to_column())) %>%
    unnest(cols = (mut_prob)) %>%
    rename(m_prob = .x, terminal = rowid)
  
  return(a)
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
    geom_smooth(
      aes(group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay))
    ) +
    facet_grid(cols = rev(vars(start_mut_prob, prob_mut_probs, gauss, remap, delay))) +
    theme(legend.position = "None")
  
  layout = 'AAAA
            AAAA
            BBBB'
  
  x = q / z + plot_annotation(title = "Average performance of solutions over evolution")
  print(x + plot_layout(guides = 'collect', design = layout))
  ggsave("fitness_over_time_per_run_across_approachs_details.pdf",
         width = 34,
         height = 15,
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

#plots the mutation probabilities of all terminals over all parroaches and on top puts plots of fitness over time
plot_mut_rates_terminals_compared_to_fit_over_time = function(a){
  best_fit = ggplot(a,
                    aes(x = gen, 
                        y = best_fit,
                        color = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                        fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)#,
                        # linetype = terminal
                    )) 
  mut_p = ggplot(a,
                 aes(x = gen, 
                     y = m_prob,
                     color = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay),
                     fill = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay)#,
                     # linetype = terminal
                 )) +
    geom_smooth() +
    facet_grid(cols = rev(vars(start_mut_prob, prob_mut_probs, gauss, remap, delay)),
               rows = vars(terminal))
  
  q = best_fit + 
    geom_line(linewidth = 0.1,
              linetype = "dotted",
              aes(          
                group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay, run))
    ) + 
    geom_smooth(
      aes(group = interaction(start_mut_prob, prob_mut_probs, gauss, remap, delay))
    ) +
    facet_grid(cols = rev(vars(start_mut_prob, prob_mut_probs, gauss, remap, delay))) +
    theme(legend.position = "None")+
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("best_fit", sep = "")) +
    theme_bw()
  
  z = mut_p +  
    facet_grid(cols = rev(vars(start_mut_prob, prob_mut_probs, gauss, remap, delay)),
               rows = vars(terminal)) +
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("mut_prob", sep = "")) +
    theme_bw() + guides(fill=guide_legend(title="approach"))
  
  
  layout = 'AAAA
          BBBB
          BBBB
          BBBB
          BBBB
          BBBB'
  
  x = q / z + plot_annotation(title = "fitness and mutation rates over evolution") + plot_layout(guides = 'collect', design = layout)
  print(x)
  return(x)
}


###test for best fitness
test_best_fitness = function(data){
  return(
    setDT(data %>% filter(gen == max(gen)))[, approach := paste(start_mut_prob, prob_mut_probs, gauss, remap, delay, sep = "_")] %>% 
      t_test(data = .,
             best_fit ~ approach,
             p.adjust.method = "BH")
  )
}

### Calculate effect size for best fitness
effects_size_best_fitness = function(data){
  return(
    setDT(data %>% filter(gen == max(gen)))[, approach := paste(start_mut_prob, prob_mut_probs, gauss, remap, delay, sep = "_")] %>% 
    cohens_d(.,
             best_fit~approach)
  )
  
}

####Priduce a latex table for p values
produce_latex_table_p_values = function(data){
  table = test_best_fitness(data) %>% 
    mutate(.y. = "Best fitness") %>% 
    mutate(
      p.adj = paste(p.adj, p.adj.signif, sep = " "),
      comparison = paste(group1,group2, sep = " - ")) %>% 
    select(-c(n1,n2,statistic,p,p.adj.signif)) %>% 
    pivot_wider(id_cols = group1, names_from = group2, values_from = "p.adj")  %>% 
    mutate(across(everything(),
                  ~ cell_spec(., "html", bold=ifelse(grepl("\\*",.), T, F)))) %>%
    kable(caption="Adjusted p-values of pairwise comparison over best-fit", 
          format = "html",
          escape = F) %>%
    kable_styling("striped", full_width = F)
  readr::write_file(table, "t_test_table.html")
  
  return(table)
}

####Priduce a latex table for effect sizes
produce_latex_table_effect_sizes = function(data){
  table = effects_size_best_fitness(data) %>% 
    mutate(.y. = "Best fitness") %>% 
    mutate(
      magnitude = ifelse(magnitude == "large","***",
                         ifelse(magnitude == "moderate", "**",
                                ifelse(magnitude == "small","*","ns"))),
      effsize = format(round(effsize, 3), nsmall = 3),
      effect_size = paste(effsize,magnitude, sep = " ")) %>% 
    select(-c(n1,n2,effsize, magnitude,.y.)) %>% 
    pivot_wider(id_cols = group1, names_from = "group2", values_from = "effect_size")  %>% 
    mutate(across(everything(),
                  ~ cell_spec(., "html", bold=ifelse(grepl("\\*",.), T, F)))) %>%
    kable(caption="Effect size of pairwise comparison over best-fit",
          format = "html",
          escape = F) %>%
    kable_styling("striped",
                  full_width = F)
  readr::write_file(table, "effect_size_table.html")
  return(table)
}


#use when need loading from raw data and not .csv
# data = read_and_save(wd)
# a = transform_psge_data_to_read_mut_probs(data)
# save(a, file = "data_gram_mut_probs.R")

data = setDT(read.csv2("data.csv", header = T, sep =",")[-1])[,best_fit :=  as.numeric(best_fit)]
load("data_gram_mut_probs.R")

fitness_over_time_across_approachs(data = data %>% subset(alg_type == "co-psge"))
fitness_over_time_across_approachs(data = data %>% subset(alg_type == "psge"))


make_best_fitness_t_test_boxplot_gen(data = data %>% subset(alg_type == "co-psge"), gen = 1000)
make_best_fitness_t_test_boxplot_gen(data = data %>% subset(alg_type == "psge"), gen = 1000)


plot_mut_rates_terminals_compared_to_fit_over_time(a %>% subset(remap == "False") %>% mutate(best_fit = as.numeric(best_fit)))

produce_latex_table_p_values(data = data %>% filter(alg_type == "psge", delay == "False"))
produce_latex_table_effect_sizes(data = data %>% filter(alg_type == "psge", delay == "False"))

