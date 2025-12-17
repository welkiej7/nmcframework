require(jsonlite)
require(dplyr)
require(ergm)
require(network)
require(statnet)
library(coda)

# Your networks should be in a json file where each key is a simulation 
# number and each value should be a list of tuples for networks

# Your opinion spaces should be in a csv file where each column is 
# named like Topic1, Topic2 etc. 


forge_interaction_network <- function(simulations, simulation_id, opinion_space){
  
  edges <- simulations$Networks$Interaction_Edges[[simulation_id]]
  edges <- lapply(edges, function(x) c(x[[1]], x[[2]]))
  edges_df <- do.call(rbind.data.frame, edges)
  
  names(edges_df) <- c("from","to")
  edges_df$from <- as.numeric(edges_df$from)
  edges_df$to <- as.numeric(edges_df$to)
  
  vertices <- sort(unique(c(edges_df$from, edges_df$to)))
  n_vertices <- length(vertices)
  
  g <- network::network.initialize(n_vertices, directed = FALSE)
  
  network::set.vertex.attribute(g, "vertex.names", vertices)
  for(i in 1:nrow(edges_df)){
    network::add.edge(g, 
                      tail = which(vertices == edges_df$from[i]),
                      head = which(vertices == edges_df$to[i]))
  }
  
  for(column in 1:length(opinion_space)){
    network::set.vertex.attribute(g, 
                                  attrname = paste("Topic", column, sep = ""), 
                                  value = opinion_space[[column]])
  }
  
  network::set.vertex.attribute(g, attrname = "Popularity", value = sna::degree(g))
  return (list(g, simulations$Params$Beta[[simulation_id]], simulations$Params$Lambda[[simulation_id]]))
  
}
forge_latent_network <- function(simulations, simulation_id, opinion_space){
  
  edges <- simulations$Networks$Latent_Edges[[simulation_id]]
  edges <- lapply(edges, function(x) c(x[[1]], x[[2]]))
  edges_df <- do.call(rbind.data.frame, edges)
  
  names(edges_df) <- c("from","to")
  edges_df$from <- as.numeric(edges_df$from)
  edges_df$to <- as.numeric(edges_df$to)
  
  vertices <- sort(unique(c(edges_df$from, edges_df$to)))
  n_vertices <- length(vertices)
  
  g <- network::network.initialize(n_vertices, directed = FALSE)
  
  network::set.vertex.attribute(g, "vertex.names", vertices)
  for(i in 1:nrow(edges_df)){
    network::add.edge(g, 
                      tail = which(vertices == edges_df$from[i]),
                      head = which(vertices == edges_df$to[i]))
  }
  
  for(column in 1:length(opinion_space)){
    network::set.vertex.attribute(g, 
                                  attrname = paste("Topic", column, sep = ""), 
                                  value = opinion_space[[column]])
  }
  
  network::set.vertex.attribute(g, attrname = "Popularity", value = sna::degree(g))
  
  return (list(g, simulations$Params$Beta[[simulation_id]], simulations$Params$Lambda[[simulation_id]]))
  
}
fit_single_model_advanced <- function(net, 
                                      thread_count = 4,
                                      MCMC.burnin = 50000,
                                      MCMC.samplesize = 10000,
                                      MCMLE.maxit = 30,
                                      decay = 0.0001,
                                      pow = 1,
                                      model_counter = 1,
                                      include_gwesp = FALSE) {
  
  run_control = control.ergm(parallel = thread_count,
                             MCMC.burnin = MCMC.burnin,
                             MCMC.samplesize = MCMC.samplesize,
                             MCMLE.maxit = MCMLE.maxit)
  
  model_name <- paste0("model.", sprintf("%02d", model_counter))
  
  formula <- net ~ edges + 
    absdiff("Topic1", pow = pow) + 
    absdiff("Topic2", pow = pow) + 
    absdiff("Topic1"):absdiff("Topic2") + 
    nodecov("Popularity") + 
    gwdegree(decay = decay, fixed = TRUE)
  
  if (include_gwesp) {
    formula <- update(formula, . ~ . + gwesp(decay = decay, fixed = TRUE))
  }
  
  result <- tryCatch({
    m <- ergm(as.formula(formula),
              constraints = ~bd(minout = 1),
              control = run_control)
    print(paste(model_name, "** PASS"))
    m$embedded_id <- model_counter
    named_model <- list()
    named_model[[model_name]] <- m
    named_model
  }, error = function(e) {
    print(paste(model_name, "** FAILED:", e$message))
    NULL
  })
  
  return(result)
}

evaluate_single_model <- function(fit, 
                                  model_name = "model.01", 
                                  nsim = 100, 
                                  thread_count = 4, 
                                  timeout = 600) {
  
  result <- tryCatch({
    withTimeout({
      if (is.null(fit) || !inherits(fit, "ergm")) {
        warning(paste("Skipping", model_name, ": Invalid model object"))
        return(NULL)
      }
      
      if (is.null(fit$sample) || length(fit$sample) == 0) {
        warning(paste("Skipping", model_name, ": No sample data available"))
        return(NULL)
      }
      
      print(paste("Processing", model_name))
      
      mcmc_obj <- as.mcmc(fit$sample[[1]])
      
      geweke_stats <- try(geweke.diag(mcmc_obj)$z, silent = TRUE)
      acf_stats <- try(apply(mcmc_obj, 2, function(x) acf(x, lag.max = 50, plot = FALSE)$acf[2]), silent = TRUE)
      ess_stats <- try(effectiveSize(mcmc_obj), silent = TRUE)
      
      mcmc_diagnostics <- list(
        geweke = if (inherits(geweke_stats, "try-error")) NULL else geweke_stats,
        autocorr = if (inherits(acf_stats, "try-error")) NULL else acf_stats,
        eff_size = if (inherits(ess_stats, "try-error")) NULL else ess_stats
      )
      
      tmp_t_stats <- tryCatch({
        apply(fit$sample[[1]], 2, function(x) mean(x) / sd(x))
      }, error = function(e) {
        warning("Error in initial t-stats:", e$message)
        NULL
      })
      
      print("Running GOF analysis...")
      gof_result <- gof(fit ~ degree + esp + distance,
                        control = control.gof.ergm(nsim = nsim, 
                                                   parallel = thread_count,
                                                   MCMC.burnin = 1000),
                        coef = fit$coefficients,
                        verbose = TRUE)
      
      if (is.null(gof_result)) {
        warning(paste("GOF analysis failed for", model_name))
        return(NULL)
      }
      
      get_tstats <- function(summary_mat, sim_mat) {
        sapply(1:nrow(summary_mat), function(j) {
          dif <- summary_mat[j, 1] - sim_mat[, j]
          if (length(dif) > 0) mean(dif, na.rm = TRUE) / sd(dif, na.rm = TRUE) else NA
        })
      }
      
      t_stats_dist <- if (!is.null(gof_result$summary.dist)) get_tstats(gof_result$summary.dist, gof_result$sim.dist) else NULL
      t_stats_espart <- if (!is.null(gof_result$summary.espart)) get_tstats(gof_result$summary.espart, gof_result$sim.espart) else NULL
      t_stats_deg <- if (!is.null(gof_result$summary.degree)) get_tstats(gof_result$summary.degree, gof_result$sim.degree) else NULL
      
      gof_t_stats <- c(t_stats_dist, t_stats_espart, t_stats_deg)
      gof_t_stats <- gof_t_stats[is.finite(gof_t_stats)]
      
      if (length(gof_t_stats) == 0) {
        warning(paste("No valid t-statistics in", model_name))
        return(NULL)
      }
      
      gof_summary <- c(
        prop_within_2sd = mean(abs(gof_t_stats) <= 2, na.rm = TRUE),
        median_abs_t = median(abs(gof_t_stats), na.rm = TRUE),
        max_abs_t = max(abs(gof_t_stats), na.rm = TRUE)
      )
      
      evaluated <- list()
      evaluated[[model_name]] <- list(
        initial_t_stats = tmp_t_stats,
        gof_result = gof_summary,
        t_stats = list(
          distance = t_stats_dist[!is.na(t_stats_dist)],
          espart = t_stats_espart[!is.na(t_stats_espart)],
          degree = t_stats_deg[!is.na(t_stats_deg)]
        ),
        mcmc_diagnostics = mcmc_diagnostics
      )
      
      print(paste("Successfully evaluated", model_name))
      return(evaluated)
      
    }, timeout = timeout)
  }, TimeoutException = function(e) {
    warning(paste("Timeout while evaluating", model_name))
    return(NULL)
  }, error = function(e) {
    warning(paste("Error in model", model_name, ":", e$message))
    return(NULL)
  })
  
  return(result)
}

simulate_without_latent <- function(fit,
                                    model_name = "model.01",
                                    nsim = 100) {
  
  coefs <- fit[[model_name]]$coefficients
  formula  <- fit[[model_name]]$formula
  
  coefs[['absdiff.Topic2']] <- 0 # Ceteris paribus. 
  coefs[['absdiff.Topic1:absdiff.Topic2']] <- 0 # Ceteris paribus.
  
  
  network <- simulate(formula, coef = coefs, nsim = nsim)
  return(network)

}

simulate_with_latent <- function(fit,
                                    model_name = "model.01",
                                    nsim = 100) {
  
  coefs <- fit[[model_name]]$coefficients
  formula  <- fit[[model_name]]$formula

  
  network <- simulate(formula, coef = coefs, nsim = nsim)
  return(network)
  
}

save_the_networks <- function(simulations, sim_save_path,
                              type = "latent"){
  edges <- list()
  for(sim in 1:length(simulations)){
    edges[[sim]] <- network::as.edgelist(simulations[[sim]])
  }
  for(network in 1:length(edges)){
    edges[[network]] <- as.data.frame(edges[[network]])
  }

  for(network in 1:length(edges)){
    readr::write_csv(edges[[network]], file = paste(sim_save_path,"/",type,'_',network,'.csv',sep = ""))
  }
}

run_latent_pipeline <- function(simulations,
                                opinion_space,
                                sim_id,
                                nsim,
                                thread_count,
                                MCMC.burnin = 50000,
                                MCMC.samplesize = 10000,
                                MCMLE.maxit = 30,
                                decay = 0.001,
                                pow = 1,
                                timeout = 600,
                                model_name = "model.01",
                                folder_name) {
  tryCatch({
    # Construct final folder path
    folder_path <- paste0(folder_name, "_", sim_id)
    if (!dir.exists(folder_path)) dir.create(folder_path, recursive = TRUE)
    
    # Forge the network
    network_data <- forge_latent_network(simulations, simulation_id = sim_id, opinion_space)
    network <- network_data[[1]]
    
    cat("Fitting the network\n")
    m <- fit_single_model_advanced(net = network, thread_count = thread_count,
                                   MCMC.burnin = MCMC.burnin,
                                   MCMC.samplesize = MCMC.samplesize,
                                   MCMLE.maxit = MCMLE.maxit,
                                   decay = decay,
                                   pow = pow,
                                   model_counter = sim_id)
    
    cat("Completed fit, evaluating...\n")
    eval_m <- evaluate_single_model(m[[model_name]],
                                    model_name = model_name,
                                    nsim = nsim,
                                    thread_count = thread_count,
                                    timeout = timeout)
    
    cat("Simulating with latent effects...\n")
    sims_wlatent <- simulate_with_latent(m, model_name = model_name, nsim = nsim)
    
    cat("Simulating without latent effects...\n")
    sims_wolatent <- simulate_without_latent(m, model_name = model_name, nsim = nsim)
    
    cat("Saving the results...\n")
    jsonlite::write_json(eval_m, file.path(folder_path, paste0(sim_id, "_evaluations.json")))
    
    save_the_networks(sims_wlatent, folder_path, type = "latent")
    save_the_networks(sims_wolatent, folder_path, type = "wolatent")
    
    return(TRUE)
  }, error = function(e) {
    return(list(success = FALSE, error = e$message))
  })
}


