# ============================================================================ #
# CAPÍTULO 4 – ESTUDIO DE SIMULACIÓN MONTE CARLO                               
# Procesos de selección de covariables en el modelo de regresión lineal           
# ============================================================================ #
#
# MODELO VERDADERO
# ----------------
# Y = beta1*X1 + ... + beta5*X5 + 0*X6 + ... + 0*Xp + e
#
# NOTA: al estandarizar X antes de ajustar, el intercepto desaparece del
# modelo estandarizado (media de X = 0 tras centrar), por lo que no se incluye.
#
# Las 5 PRIMERAS covariables son siempre las relevantes.
# Covariables: Xj ~ N(0, sd=5) (independientes en IND, dependientes en DEP)
# Error:       epsilon ~ N(0, sd=1)
#
# CONFIGURACIONES DE BETAS (ordenados de mayor a menor magnitud)
# --------------------------------------------------------------
# Configuración I  (señal fuerte):
#   beta = c(1.0, -0.75, 0.6, 0.5, -0.4) * 5
#
# Configuración II (señal débil):
#   beta = c(1.0, -0.75, 0.6, 0.5, -0.4)
#
# ESCENARIOS
# ----------
# Cada escenario combina:
#   - tipo: IND (covariables independientes) o DEP (con dependencia)
#   - config: I (señal fuerte) o II (señal débil)
#   - bloque dimensional: n>p (p=50, n en {100,200,500})
#                         p>n (p=100, n en {20,50})
# Nombre del escenario (texto): IND I, IND II, DEP I, DEP II
#
# En el bloque n>p todos los métodos son aplicables
# En el bloque p>n el estimador OLS no está definido y solo
# se comparan LASSO, Adaptive LASSO, SCAD y Boosting lineal.
#
# ESTRUCTURA DEL SCRIPT
# --------------------------------------------
# FASE 1 – Simulación: para cada escenario y método, guarda una matriz de
#          coeficientes estimados [M x p] y el MSE de predicción por réplica.
# FASE 2 – Métricas: calcula todas las métricas a partir de las matrices
#          guardadas en la Fase 1. Es prácticamente instantánea y se puede
#          repetir sin reejecutar las simulaciones.
#
# MÉTRICAS (promedio sobre M réplicas)
# --------------------------------------
# TPR       : prop. de variables relevantes correctamente seleccionadas
# FPR       : prop. de variables ruido incorrectamente seleccionadas
# Exactitud : prop. de réplicas con selección exactamente correcta
# MSE_est   : ||beta_hat - beta_true||_2^2 promediado sobre réplicas,
#             calculado sobre todos los coeficientes (relevantes y ruido)
# MSE_pred  : error cuadrático medio de predicción sobre datos de test
# TPR_Xj    : prop. de réplicas en que la variable Xj fue seleccionada
#             (j = 1,...,p). Usasa en los heatmaps de selección.
#
# NOTA SOBRE ESTANDARIZACIÓN
# ---------------------------
# Antes de aplicar cualquier método, X se estandariza (media 0, sd 1) para
# que todos los métodos operen en la misma escala y la comparación sea justa.
# Se desactiva la estandarización interna de cada librería para que todos
# reciban exactamente los mismos datos. Los coeficientes se desescalan al
# final para compararlos con beta_true.
#
# PAQUETES: install.packages(c("glmnet","ncvreg","leaps","mboost","xtable","tictoc"))
# ============================================================================ #

library(glmnet)
library(ncvreg)
library(leaps)
library(mboost)
library(xtable)
library(tictoc)

# ============================================================================ #
# 0. PARÁMETROS GLOBALES ------
# ============================================================================ #

M          <- 1000    # réplicas Monte Carlo
N_REL      <- 5       # variables relevantes: siempre las 5 primeras
SIGMA      <- 1       # desviación típica del error
SD_X       <- 5       # desviación típica de las covariables (> SIGMA)
N_TEST     <- 500     # observaciones de test para MSE_pred
RHO_DEP    <- 0.6     # correlación para los escenarios DEP (AR(1))

# Betas ordenados de mayor a menor magnitud absoluta
BETA_BASE  <- c(1.0, -0.75, 0.6, 0.5, -0.4)
BETA_CFG1  <- BETA_BASE * 5    # Configuración I: señal fuerte
BETA_CFG2  <- BETA_BASE        # Configuración II: señal débil

# Orden de métodos: se aplica en TODAS las tablas y gráficas
ORDEN_METODOS <- c("Forward", "Backward", "Stepwise",
                   "LASSO", "Ad.LASSO", "SCAD", "Boosting")

DIR_OUT      <- "Scipt_Junio"
GUARDAR_CADA <- 1000

if (!dir.exists(DIR_OUT)) dir.create(DIR_OUT, recursive = TRUE)

# ============================================================================ #
# 1. ESCENARIOS ------
# ============================================================================ #
# Cada escenario combina: tipo de dependencia (IND/DEP) x configuración (I/II)
# x bloque dimensional (n>p, p>n). El nombre del escenario sigue la notación
# IND I, IND II, DEP I, DEP II que usamos en el texto.

bloque_base <- data.frame(
  bloque = c(rep("n>p", 6), rep("p>n", 4)),
  p      = c(rep(50,  6),   rep(100, 4)),
  n      = c(rep(c(100, 200, 500), each = 2),
             rep(c(20, 50),        each = 2)),
  config = rep(c("I", "II"), times = 5),
  stringsAsFactors = FALSE
)

escenarios <- rbind(
  cbind(tipo = "IND", bloque_base),
  cbind(tipo = "DEP", bloque_base)
)
escenarios$nombre_escenario <- paste0(escenarios$tipo, " ", escenarios$config)
rownames(escenarios) <- NULL
n_esc <- nrow(escenarios)

METODOS_NP <- c("Forward", "Backward", "Stepwise",
                "LASSO", "Ad.LASSO", "SCAD", "Boosting")
METODOS_PN <- c("LASSO", "Ad.LASSO", "SCAD", "Boosting")

# Métricas globales que se calculan por método y escenario
METRICAS <- c("TPR", "FPR", "Exactitud", "MSE_est", "MSE_pred")

cat("=============================================================\n")
cat("ESTUDIO MONTE CARLO - CAPÍTULO 4\n")
cat(sprintf("Escenarios: %d | Réplicas: %d\n", n_esc, M))
cat("=============================================================\n\n")
print(escenarios); cat("\n")


# ============================================================================ #
# 2. FUNCIONES AUXILIARES ------
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# 2a. Generar datos
# ---------------------------------------------------------------------------- #
# Genera X con sd=SD_X y epsilon con sd=SIGMA. En los escenarios IND las
# covariables son independientes; en los escenarios DEP siguen una estructura
# de correlación Toeplitz con parámetro RHO_DEP, escalada para mantener sd=SD_X.
# Devuelve X e Y de entrenamiento, más X_test e Y_test para MSE_pred.
generar_datos <- function(n, p, beta_rel, seed, tipo = "IND") {
  
  beta_true <- c(beta_rel, rep(0, p - N_REL))
  
  # Matriz de correlación
  if (tipo == "IND") {
    Sigma <- diag(p)
  } else {
    # Estructura Toeplitz: cor(Xi, Xj) = RHO_DEP^|i-j|
    idx   <- 1:p
    Sigma <- RHO_DEP ^ abs(outer(idx, idx, "-"))
  }
  L <- chol(Sigma) * SD_X   # escalar para que Var(Xj) = SD_X^2
  
  set.seed(seed)
  Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  X <- Z %*% L
  colnames(X) <- paste0("X", 1:p)
  Y <- X %*% beta_true + rnorm(n, 0, SIGMA)
  
  # Conjunto de test: misma estructura, semilla desplazada para independencia
  set.seed(seed + 99999)
  Z_test <- matrix(rnorm(N_TEST * p), nrow = N_TEST, ncol = p)
  X_test <- Z_test %*% L
  colnames(X_test) <- paste0("X", 1:p)
  Y_test <- X_test %*% beta_true + rnorm(N_TEST, 0, SIGMA)
  
  list(X = X, Y = as.numeric(Y),
       X_test = X_test, Y_test = as.numeric(Y_test),
       beta_true = beta_true)
}

# ---------------------------------------------------------------------------- #
# 2b. Estandarizar X y desescalar coeficientes
# ---------------------------------------------------------------------------- #
estandarizar_X <- function(X) {
  med_X <- colMeans(X)
  sd_X  <- apply(X, 2, sd)
  sd_X[sd_X == 0] <- 1
  X_std <- scale(X, center = med_X, scale = sd_X)
  list(X_std = X_std, med_X = med_X, sd_X = sd_X)
}

estandarizar_X_test <- function(X_test, med_X, sd_X) {
  scale(X_test, center = med_X, scale = sd_X)
}

desescalar_coefs <- function(coef_std, sd_X) {
  coef_std / sd_X
}

# ---------------------------------------------------------------------------- #
# 2c. Forward / Backward / Stepwise (BIC vía regsubsets)
# ---------------------------------------------------------------------------- #
aplicar_regsubsets <- function(X_std, Y, sd_X, metodo) {
  p  <- ncol(X_std)
  df <- as.data.frame(cbind(Y = Y, X_std))
  ft <- regsubsets(Y ~ ., data = df, nvmax = p, method = metodo)
  rs <- summary(ft)
  k  <- which.min(rs$bic)
  sel <- as.integer(rs$which[k, -1])
  
  vars_sel <- paste0("X", which(sel == 1))
  ce_std <- rep(0, p)
  if (length(vars_sel) > 0) {
    fm  <- as.formula(paste("Y ~", paste(vars_sel, collapse = "+")))
    ftl <- lm(fm, data = df)
    idx <- as.integer(sub("X", "", names(coef(ftl))[-1]))
    ce_std[idx] <- coef(ftl)[-1]
  }
  ce <- desescalar_coefs(ce_std, sd_X)
  list(sel = sel, coef_est = as.numeric(ce), coef_std = as.numeric(ce_std))
}

# ---------------------------------------------------------------------------- #
# 2d. LASSO (CV 10-fold, estandarización ya hecha externamente)
# ---------------------------------------------------------------------------- #
aplicar_lasso <- function(X_std, Y, sd_X) {
  cv     <- cv.glmnet(X_std, Y, alpha = 1, nfolds = 10, standardize = FALSE)
  ft     <- glmnet(X_std, Y, alpha = 1, lambda = cv$lambda.min,
                   standardize = FALSE)
  ce_std <- as.numeric(coef(ft))[-1]
  ce     <- desescalar_coefs(ce_std, sd_X)
  list(sel = as.integer(ce_std != 0), coef_est = ce, coef_std = ce_std)
}

# ---------------------------------------------------------------------------- #
# 2e. Adaptive LASSO (Ridge inicial + LASSO ponderado, CV 10-fold)
# ---------------------------------------------------------------------------- #
aplicar_alasso <- function(X_std, Y, sd_X) {
  rc     <- cv.glmnet(X_std, Y, alpha = 0, nfolds = 10, standardize = FALSE)
  br     <- as.numeric(coef(rc, s = rc$lambda.min))[-1]
  pw     <- 1 / (abs(br) + 1e-6)
  cv     <- cv.glmnet(X_std, Y, alpha = 1, penalty.factor = pw,
                      nfolds = 10, standardize = FALSE)
  ft     <- glmnet(X_std, Y, alpha = 1, lambda = cv$lambda.min,
                   penalty.factor = pw, standardize = FALSE)
  ce_std <- as.numeric(coef(ft))[-1]
  ce     <- desescalar_coefs(ce_std, sd_X)
  list(sel = as.integer(ce_std != 0), coef_est = ce, coef_std = ce_std)
}

# ---------------------------------------------------------------------------- #
# 2f. SCAD (CV 10-fold, sin estandarización interna)
# ---------------------------------------------------------------------------- #
aplicar_scad <- function(X_std, Y, sd_X) {
  cv     <- cv.ncvreg(X_std, Y, penalty = "SCAD", nfolds = 10)
  cf     <- coef(cv$fit, lambda = cv$lambda.min)
  ce_std <- cf[-1]
  ce     <- desescalar_coefs(ce_std, sd_X)
  list(sel = as.integer(ce_std != 0), coef_est = as.numeric(ce),
       coef_std = as.numeric(ce_std))
}

# ---------------------------------------------------------------------------- #
# 2g. Boosting lineal (L2-boosting, early stopping por CV 10-fold)
# ---------------------------------------------------------------------------- #
aplicar_boosting <- function(X_std, Y, sd_X) {
  p  <- ncol(X_std)
  df <- as.data.frame(cbind(Y = Y, X_std))
  ft <- glmboost(Y ~ ., data = df,
                 control = boost_control(mstop = 500, nu = 0.1))
  cv <- cvrisk(ft,
               folds = cv(model.weights(ft), type = "kfold", B = 10),
               grid  = seq(10, 500, by = 10))
  fo <- glmboost(Y ~ ., data = df,
                 control = boost_control(mstop = mstop(cv), nu = 0.1))
  cc      <- coef(fo)
  cv2     <- cc[names(cc) != "(Intercept)"]
  ce_std  <- rep(0, p); names(ce_std) <- paste0("X", 1:p)
  for (nm in names(cv2)) if (nm %in% names(ce_std)) ce_std[nm] <- cv2[nm]
  ce <- desescalar_coefs(as.numeric(ce_std), sd_X)
  list(sel = as.integer(abs(ce_std) > 1e-6), coef_est = as.numeric(ce),
       coef_std = as.numeric(ce_std))
}

# ---------------------------------------------------------------------------- #
# 2h. Función principal de simulación de un escenario
# ---------------------------------------------------------------------------- #
# Guarda dos objetos por escenario:
#   betas_lista : lista de matrices [M x p] de coeficientes (escala original)
#   mse_lista   : lista de vectores [M] con MSE_pred de cada réplica
simular_escenario <- function(esc) {
  
  tipo_e   <- escenarios$tipo[esc]
  bloque_e <- escenarios$bloque[esc]
  p_e      <- escenarios$p[esc]
  n_e      <- escenarios$n[esc]
  config_e <- escenarios$config[esc]
  beta_rel <- if (config_e == "I") BETA_CFG1 else BETA_CFG2
  mets_e   <- if (bloque_e == "n>p") METODOS_NP else METODOS_PN
  
  cat(sprintf("\n[Escenario %d/%d] %s | %s | p=%d | n=%d | Config. %s\n",
              esc, n_esc, tipo_e, bloque_e, p_e, n_e, config_e))
  cat(sprintf("  Betas relevantes: %s\n",
              paste(round(beta_rel, 3), collapse = ", ")))
  
  betas_lista <- lapply(mets_e, function(m) {
    mat <- matrix(NA_real_, nrow = M, ncol = p_e)
    colnames(mat) <- paste0("X", 1:p_e)
    mat
  })
  names(betas_lista) <- mets_e
  
  mse_lista <- lapply(mets_e, function(m) rep(NA_real_, M))
  names(mse_lista) <- mets_e
  
  pb <- txtProgressBar(min = 0, max = M, style = 3)
  tic(sprintf("Escenario %d", esc))
  
  for (i in seq_len(M)) {
    
    seed_i  <- 10000 * esc + i
    datos_i <- generar_datos(n_e, p_e, beta_rel, seed_i, tipo = tipo_e)
    
    std_i      <- estandarizar_X(datos_i$X)
    X_std_i    <- std_i$X_std
    sd_X_i     <- std_i$sd_X
    med_X_i    <- std_i$med_X
    X_test_std <- estandarizar_X_test(datos_i$X_test, med_X_i, sd_X_i)
    
    for (met in mets_e) {
      r <- tryCatch({
        switch(met,
               "Forward"  = aplicar_regsubsets(X_std_i, datos_i$Y, sd_X_i, "forward"),
               "Backward" = aplicar_regsubsets(X_std_i, datos_i$Y, sd_X_i, "backward"),
               "Stepwise" = aplicar_regsubsets(X_std_i, datos_i$Y, sd_X_i, "seqrep"),
               "LASSO"    = aplicar_lasso(X_std_i, datos_i$Y, sd_X_i),
               "Ad.LASSO" = aplicar_alasso(X_std_i, datos_i$Y, sd_X_i),
               "SCAD"     = aplicar_scad(X_std_i, datos_i$Y, sd_X_i),
               "Boosting" = aplicar_boosting(X_std_i, datos_i$Y, sd_X_i))
      }, error = function(e) NULL)
      
      if (!is.null(r)) {
        betas_lista[[met]][i, ] <- r$coef_est
        
        Y_pred <- X_test_std %*% r$coef_std
        mse_lista[[met]][i] <- mean((datos_i$Y_test - Y_pred)^2)
      }
    }
    
    setTxtProgressBar(pb, i)
    
    if (i %% GUARDAR_CADA == 0) {
      saveRDS(list(betas = betas_lista, mse = mse_lista),
              file.path(DIR_OUT,
                        sprintf("betas_esc%02d_rep%04d.rds", esc, i)))
    }
  }
  
  close(pb)
  toc()
  
  fname_final <- file.path(DIR_OUT, sprintf("betas_esc%02d_FINAL.rds", esc))
  saveRDS(list(betas = betas_lista, mse = mse_lista), fname_final)
  cat(sprintf("  -> Guardado: %s\n", fname_final))
  
  invisible(list(betas = betas_lista, mse = mse_lista))
}

# ---------------------------------------------------------------------------- #
# 2i. Calcular métricas a partir de los objetos guardados en Fase 1
# ---------------------------------------------------------------------------- #
# Recibe la lista de matrices de betas [M x p], el vector MSE_pred y el
# vector beta_true, y devuelve un data.frame con las métricas promediadas
# sobre M réplicas. Es independiente de la simulación: se puede llamar en
# cualquier momento sobre los archivos .rds guardados.
calcular_metricas_desde_betas <- function(betas_lista, mse_lista, beta_true, p) {
  
  idx_rel <- 1:N_REL
  idx_rui <- (N_REL + 1):p
  sel_v   <- as.integer(beta_true != 0)
  
  resultado <- lapply(names(betas_lista), function(met) {
    
    mat     <- betas_lista[[met]]   # M x p
    mse_vec <- mse_lista[[met]]     # vector M
    
    # Vector de selección para cada réplica: 1 si coef supera umbral, 0 si no
    # umbral mínimo para ignorar coeficientes residuales
    UMBRAL_SEL <- 1e-6
    sel_mat <- (abs(mat) > UMBRAL_SEL) * 1L
    sel_mat[is.na(mat)] <- NA
    
    # TPR
    TPR <- mean(rowMeans(sel_mat[, idx_rel, drop = FALSE], na.rm = TRUE),
                na.rm = TRUE)
    
    # FPR
    FPR <- mean(rowMeans(sel_mat[, idx_rui, drop = FALSE], na.rm = TRUE),
                na.rm = TRUE)
    
    # Exactitud
    exacta <- apply(sel_mat, 1, function(x) {
      if (any(is.na(x))) return(NA)
      as.numeric(all(x == sel_v))
    })
    Exactitud <- mean(exacta, na.rm = TRUE)
    
    # MSE de estimación: ||beta_hat - beta_true||^2 sobre todos los coefs
    dif_todos <- sweep(mat, 2, beta_true, "-")
    MSE_est   <- mean(rowSums(dif_todos^2, na.rm = TRUE), na.rm = TRUE)
    
    # MSE de predicción: media sobre las M réplicas
    MSE_pred <- mean(mse_vec, na.rm = TRUE)
    
    # TPR individual para todas las variables (X1..Xp): heatmaps
    tpr_all <- colMeans(sel_mat, na.rm = TRUE)
    names(tpr_all) <- paste0("TPR_X", 1:p)
    
    c(Metodo    = met,
      TPR       = round(TPR,       4),
      FPR       = round(FPR,       4),
      Exactitud = round(Exactitud, 4),
      MSE_est   = round(MSE_est,   4),
      MSE_pred  = round(MSE_pred,  4),
      round(tpr_all, 4))
  })
  
  df <- as.data.frame(do.call(rbind, resultado), stringsAsFactors = FALSE)
  cols_num <- setdiff(names(df), "Metodo")
  df[cols_num] <- lapply(df[cols_num], as.numeric)
  df
}

# ---------------------------------------------------------------------------- #
# 2j. Reordenar métodos según ORDEN_METODOS
# ---------------------------------------------------------------------------- #
ordenar_metodos <- function(df) {
  orden_presentes <- ORDEN_METODOS[ORDEN_METODOS %in% df$Metodo]
  df[match(orden_presentes, df$Metodo), ]
}

# ---------------------------------------------------------------------------- #
# 2k. Escribir tabla LaTeX directamente con celdas coloreadas
# ---------------------------------------------------------------------------- #
# Argumentos:
#   sub       : data.frame con los datos del escenario (ya ordenado)
#   n_rui     : número de variables ruido
#   caption   : caption de la tabla
#   label     : label de la tabla
#   fichero   : ruta del archivo .tex de salida
escribir_tabla_latex <- function(sub, n_rui, caption, label, fichero) {
  
  rel    <- sub$TPR * N_REL
  rui    <- sub$FPR * n_rui
  exa    <- sub$Exactitud
  mse_e  <- sub$MSE_est
  mse_p  <- sub$MSE_pred
  mets   <- sub$Metodo
  
  fmt_cel <- function(vals, es_mejor, dec) {
    fmt <- sprintf("%%.%df", dec)
    cel <- sprintf(fmt, vals)
    cel[es_mejor] <- paste0("\\cellcolor{green!25}", cel[es_mejor])
    cel
  }
  
  mejor_rel   <- rel   == max(rel,   na.rm = TRUE)
  mejor_rui   <- rui   == min(rui,   na.rm = TRUE)
  mejor_exa   <- exa   == max(exa,   na.rm = TRUE)
  mejor_mse_e <- mse_e == min(mse_e, na.rm = TRUE)
  mejor_mse_p <- mse_p == min(mse_p, na.rm = TRUE)
  
  c_rel   <- fmt_cel(rel,   mejor_rel,   2)
  c_rui   <- fmt_cel(rui,   mejor_rui,   2)
  c_exa   <- fmt_cel(exa,   mejor_exa,   3)
  c_mse_e <- fmt_cel(mse_e, mejor_mse_e, 3)
  c_mse_p <- fmt_cel(mse_p, mejor_mse_p, 4)
  
  sink(fichero)
  cat("\\begin{table}[H]\n")
  cat("\\centering\\footnotesize\n")
  cat("\\begin{tabular}{lccccc}\n")
  cat("\\toprule\n")
  cat("M\\'etodo & TPR & FPR & Exactitud & MSE~est. & MSE~pred. \\\\\n")
  cat("\\midrule\n")
  for (k in seq_along(mets)) {
    cat(sprintf("%s & %s & %s & %s & %s & %s \\\\\n",
                mets[k], c_rel[k], c_rui[k],
                c_exa[k], c_mse_e[k], c_mse_p[k]))
  }
  cat("\\bottomrule\n")
  cat("\\end{tabular}\n")
  cat(sprintf("\\caption{%s}\n", caption))
  cat(sprintf("\\label{%s}\n", label))
  cat("\\end{table}\n")
  sink()
}


# ============================================================================ #
# 3. FASE 1 – SIMULACIÓN ------
# ============================================================================ #
# Ejecuta la simulación para todos los escenarios. Esta fase es la
# computacionalmente costosa; sus resultados se guardan en disco para que
# la Fase 2 (cálculo de métricas) pueda repetirse sin volver a simular.

resultados_sim <- vector("list", n_esc)

for (esc in seq_len(n_esc)) {
  resultados_sim[[esc]] <- simular_escenario(esc)
}

cat("\n>> Fase 1 (simulación) completada.\n\n")


# ============================================================================ #
# 4. FASE 2 – CÁLCULO DE MÉTRICAS ------
# ============================================================================ #
# A partir de las matrices de betas y MSE de predicción guardadas en la
# Fase 1, se calculan las métricas para cada escenario y método.

resultados_lista <- vector("list", n_esc)

for (esc in seq_len(n_esc)) {
  
  fname <- file.path(DIR_OUT, sprintf("betas_esc%02d_FINAL.rds", esc))
  obj   <- readRDS(fname)
  
  p_e      <- escenarios$p[esc]
  config_e <- escenarios$config[esc]
  beta_rel <- if (config_e == "I") BETA_CFG1 else BETA_CFG2
  beta_true <- c(beta_rel, rep(0, p_e - N_REL))
  
  met_df <- calcular_metricas_desde_betas(obj$betas, obj$mse, beta_true, p_e)
  
  met_df$tipo   <- escenarios$tipo[esc]
  met_df$bloque <- escenarios$bloque[esc]
  met_df$p      <- p_e
  met_df$n      <- escenarios$n[esc]
  met_df$config <- config_e
  met_df$nombre_escenario <- escenarios$nombre_escenario[esc]
  
  resultados_lista[[esc]] <- met_df
}

# Alinear columnas antes de combinar: los escenarios con p=50 tienen columnas
# TPR_X1..TPR_X50 y los de p=100 tienen TPR_X1..TPR_X100. 

resultados_lista_nonnull <- Filter(Negate(is.null), resultados_lista)
todas_cols <- Reduce(union, lapply(resultados_lista_nonnull, names))
resultados_lista_alin <- lapply(resultados_lista_nonnull, function(df) {
  cols_faltantes <- setdiff(todas_cols, names(df))
  if (length(cols_faltantes) > 0) {
    df[cols_faltantes] <- NA_real_
  }
  df[, todas_cols]   # mismo orden de columnas en todos
})

resultados_df <- do.call(rbind, resultados_lista_alin)
cat("\n>> Métricas calculadas correctamente.\n\n")


# ============================================================================ #
# 5. TABLAS EN CONSOLA ------
# ============================================================================ #

cat("=== RESULTADOS POR ESCENARIO ===\n\n")

for (esc in seq_len(n_esc)) {
  
  tipo_e   <- escenarios$tipo[esc]
  bloque_e <- escenarios$bloque[esc]
  p_e      <- escenarios$p[esc]
  n_e      <- escenarios$n[esc]
  config_e <- escenarios$config[esc]
  n_rui    <- p_e - N_REL
  beta_ref <- if (config_e == "I") BETA_CFG1 else BETA_CFG2
  
  sub <- subset(resultados_df,
                tipo == tipo_e & bloque == bloque_e &
                  p == p_e & n == n_e & config == config_e)
  sub <- ordenar_metodos(sub)
  
  cat(sprintf("--- %s | %s | p=%d (%d rel + %d ruido) | n=%d | Config. %s ---\n",
              tipo_e, bloque_e, p_e, N_REL, n_rui, n_e, config_e))
  cat(sprintf("    Betas: %s\n", paste(round(beta_ref, 3), collapse = ", ")))
  
  tp <- data.frame(
    Metodo    = sub$Metodo,
    TPR       = round(sub$TPR * N_REL, 2),
    FPR       = round(sub$FPR * n_rui, 2),
    Exactitud = round(sub$Exactitud,   3),
    MSE_est   = round(sub$MSE_est,     3),
    MSE_pred  = round(sub$MSE_pred,    4),
    check.names = FALSE
  )
  print(tp, row.names = FALSE)
  
  cat(sprintf("\n  Detección por variable relevante:\n"))
  cat(sprintf("  (betas: %s)\n",
              paste(paste0("X", 1:N_REL, "=", round(beta_ref, 3)), collapse = " ")))
  tpr_v <- data.frame(Metodo = sub$Metodo)
  for (j in 1:N_REL) tpr_v[[paste0("X", j)]] <- round(sub[[paste0("TPR_X", j)]], 3)
  print(tpr_v, row.names = FALSE)
  cat(strrep("-", 72), "\n\n")
}


# ============================================================================ #
# 6. TABLAS LATEX ------
# ============================================================================ #
#
# Se generan dos tipos de tablas:
#   a) Tabla por escenario individual 
#   b) Tabla unificada por tipo, bloque y configuración
#      agrupa todos los valores de n en una sola tabla
#
# Todas las tablas destacan el mejor valor de cada columna con \cellcolor{green!25}
# Requiere \usepackage{colortbl} en el preámbulo de Overleaf.

# --- 6a. Tablas individuales por escenario ---

for (esc in seq_len(n_esc)) {
  
  tipo_e   <- escenarios$tipo[esc]
  bloque_e <- escenarios$bloque[esc]
  p_e      <- escenarios$p[esc]
  n_e      <- escenarios$n[esc]
  config_e <- escenarios$config[esc]
  n_rui    <- p_e - N_REL
  beta_ref <- if (config_e == "I") BETA_CFG1 else BETA_CFG2
  
  sub <- subset(resultados_df,
                tipo == tipo_e & bloque == bloque_e &
                  p == p_e & n == n_e & config == config_e)
  sub <- ordenar_metodos(sub)
  
  fname <- sprintf("tabla_%s_%s_p%d_n%d_cfg%s.tex",
                   tolower(tipo_e), gsub(">", "", bloque_e), p_e, n_e, config_e)
  
  escribir_tabla_latex(
    sub     = sub,
    n_rui   = n_rui,
    caption = sprintf(
      "Resultados: escenario %s (Configuración~%s), bloque $%s$, $p=%d$ (%d relevantes + %d ruido), $n=%d$. Betas: %s. La celda verde indica el mejor valor de la columna.",
      escenarios$nombre_escenario[esc], config_e, bloque_e, p_e, N_REL, n_rui, n_e,
      paste(round(beta_ref, 2), collapse = ", ")),
    label   = sprintf("tab:C4:%s_%s_p%d_n%d_cfg%s",
                      tolower(tipo_e), gsub(">", "", bloque_e), p_e, n_e, config_e),
    fichero = file.path(DIR_OUT, fname)
  )
}

# --- 6b. Tablas unificadas por tipo, bloque y configuración ---
# Una tabla agrupa todos los valores de n, con secciones separadas.
# El color verde marca el mejor valor dentro de cada sección de n.

for (tipo_g in c("IND", "DEP")) {
  for (bloque_g in c("n>p", "p>n")) {
    for (config_g in c("I", "II")) {
      
      sub_g <- subset(resultados_df,
                      tipo == tipo_g & bloque == bloque_g & config == config_g)
      if (nrow(sub_g) == 0) next
      
      p_g    <- unique(sub_g$p)
      n_vals <- sort(unique(sub_g$n))
      n_rui  <- p_g - N_REL
      beta_ref <- if (config_g == "I") BETA_CFG1 else BETA_CFG2
      
      fname_u <- sprintf("tabla_unif_%s_%s_cfg%s.tex",
                         tolower(tipo_g), gsub(">", "", bloque_g), config_g)
      
      sink(file.path(DIR_OUT, fname_u))
      cat("\\begin{table}[H]\n")
      cat("\\centering\\footnotesize\n")
      cat("\\begin{tabular}{lccccc}\n")
      cat("\\toprule\n")
      cat("M\\'etodo & TPR & FPR & Exactitud & MSE~est. & MSE~pred. \\\\\n")
      
      for (n_g in n_vals) {
        sub_n <- subset(sub_g, n == n_g)
        sub_n <- ordenar_metodos(sub_n)
        
        cat("\\midrule\n")
        cat(sprintf("\\multicolumn{6}{l}{\\textit{$n = %d$}} \\\\\n", n_g))
        cat("\\midrule\n")
        
        rel   <- sub_n$TPR * N_REL
        rui   <- sub_n$FPR * n_rui
        exa   <- sub_n$Exactitud
        mse_e <- sub_n$MSE_est
        mse_p <- sub_n$MSE_pred
        
        fmt_cel <- function(vals, es_mejor, dec) {
          fmt <- sprintf("%%.%df", dec)
          cel <- sprintf(fmt, vals)
          cel[es_mejor] <- paste0("\\cellcolor{green!25}", cel[es_mejor])
          cel
        }
        
        c_rel   <- fmt_cel(rel,   rel   == max(rel,   na.rm = TRUE), 2)
        c_rui   <- fmt_cel(rui,   rui   == min(rui,   na.rm = TRUE), 2)
        c_exa   <- fmt_cel(exa,   exa   == max(exa,   na.rm = TRUE), 3)
        c_mse_e <- fmt_cel(mse_e, mse_e == min(mse_e, na.rm = TRUE), 3)
        c_mse_p <- fmt_cel(mse_p, mse_p == min(mse_p, na.rm = TRUE), 4)
        
        for (k in seq_len(nrow(sub_n))) {
          cat(sprintf("%s & %s & %s & %s & %s & %s \\\\\n",
                      sub_n$Metodo[k],
                      c_rel[k], c_rui[k],
                      c_exa[k], c_mse_e[k], c_mse_p[k]))
        }
      }
      
      cat("\\bottomrule\n")
      cat("\\end{tabular}\n")
      cat(sprintf(
        "\\caption{Resultados del escenario %s (Configuración~%s), bloque $%s$, $p=%d$ (%d relevantes + %d ruido). Betas: %s. La celda verde indica el mejor valor de la columna dentro de cada sub-tabla.}\n",
        paste0(tipo_g, " ", config_g), config_g, bloque_g, p_g, N_REL, n_rui,
        paste(round(beta_ref, 2), collapse = ", ")))
      cat(sprintf("\\label{tab:C4:unif_%s_%s_cfg%s}\n",
                  tolower(tipo_g), gsub(">", "", bloque_g), config_g))
      cat("\\end{table}\n")
      sink()
    }
  }
}

cat(">> Tablas LaTeX exportadas en:", DIR_OUT, "\n\n")

# ============================================================================ #
# 6c. TABLAS MAESTRAS POR ESCENARIO (IND I / IND II / DEP I / DEP II) --------
# ============================================================================ #
# Una tabla por combinación tipo x config, con TODAS las filas (n,p)
# del estudio (tanto n>p como p>n), añadiendo una columna "(n,p)".

for (tipo_g in c("IND", "DEP")) {
  for (config_g in c("I", "II")) {
    
    sub_g <- subset(resultados_df, tipo == tipo_g & config == config_g)
    if (nrow(sub_g) == 0) next
    
    beta_ref <- if (config_g == "I") BETA_CFG1 else BETA_CFG2
    nombre_esc <- paste0(tipo_g, " ", config_g)
    
    fname_m <- sprintf("tabla_maestra_%s_cfg%s.tex",
                       tolower(tipo_g), config_g)
    
    sink(file.path(DIR_OUT, fname_m))
    cat("\\begin{table}[H]\n")
    cat("\\centering\\footnotesize\n")
    cat("\\begin{tabular}{llccccc}\n")
    cat("\\toprule\n")
    cat("$(n,p)$ & M\\'etodo & TPR & FPR & Exactitud & MSE~est. & MSE~pred. \\\\\n")
    
    # Orden: primero n>p (n creciente), luego p>n (n creciente)
    combs <- unique(sub_g[, c("bloque", "n", "p")])
    combs <- combs[order(factor(combs$bloque, levels = c("n>p", "p>n")), combs$n), ]
    
    for (k in seq_len(nrow(combs))) {
      bloque_k <- combs$bloque[k]
      n_k      <- combs$n[k]
      p_k      <- combs$p[k]
      n_rui_k  <- p_k - N_REL
      
      sub_n <- subset(sub_g, bloque == bloque_k & n == n_k & p == p_k)
      sub_n <- ordenar_metodos(sub_n)
      
      cat("\\midrule\n")
      
      rel   <- sub_n$TPR * N_REL
      rui   <- sub_n$FPR * n_rui_k
      exa   <- sub_n$Exactitud
      mse_e <- sub_n$MSE_est
      mse_p <- sub_n$MSE_pred
      
      fmt_cel <- function(vals, es_mejor, dec) {
        fmt <- sprintf("%%.%df", dec)
        cel <- sprintf(fmt, vals)
        cel[es_mejor] <- paste0("\\cellcolor{green!25}", cel[es_mejor])
        cel
      }
      
      c_rel   <- fmt_cel(rel,   rel   == max(rel,   na.rm = TRUE), 2)
      c_rui   <- fmt_cel(rui,   rui   == min(rui,   na.rm = TRUE), 2)
      c_exa   <- fmt_cel(exa,   exa   == max(exa,   na.rm = TRUE), 3)
      c_mse_e <- fmt_cel(mse_e, mse_e == min(mse_e, na.rm = TRUE), 3)
      c_mse_p <- fmt_cel(mse_p, mse_p == min(mse_p, na.rm = TRUE), 4)
      
      for (j in seq_len(nrow(sub_n))) {
        cat(sprintf("%s & %s & %s & %s & %s & %s & %s \\\\\n",
                    ifelse(j == 1, sprintf("$(%d,%d)$", n_k, p_k), ""),
                    sub_n$Metodo[j],
                    c_rel[j], c_rui[j],
                    c_exa[j], c_mse_e[j], c_mse_p[j]))
      }
    }
    
    cat("\\bottomrule\n")
    cat("\\end{tabular}\n")
    cat(sprintf(
      "\\caption{Resultados del escenario %s (Configuración~%s; betas: %s) para todas las combinaciones $(n,p)$ consideradas. La celda verde indica el mejor valor de la columna dentro de cada bloque $(n,p)$.}\n",
      nombre_esc, config_g, paste(round(beta_ref, 2), collapse = ", ")))
    cat(sprintf("\\label{tab:C4:maestra_%s_cfg%s}\n", tolower(tipo_g), config_g))
    cat("\\end{table}\n")
    sink()
  }
}

# ============================================================================ #
# 7. HEATMAPS DE SELECCIÓN ------
# ============================================================================ #
#
# Para cada combinación (tipo, bloque, config, n) genera un heatmap donde:
#   - Filas    = métodos (en ORDEN_METODOS)
#   - Columnas = TODAS las covariables (X1..Xp)
#   - Color    = proporción de réplicas en que la variable fue seleccionada
#                Escala continua de blanco (0) a verde oscuro (1)
#   - Una línea vertical entre X5 y X6 separa relevantes de ruido
#   - Termómetro de color como leyenda lateral (no se imprimen valores
#     numéricos en cada celda, ya que con p grande resultaría ilegible)

guardar_pdf <- function(nombre, ancho = 10, alto = 5) {
  pdf(file.path(DIR_OUT, nombre), width = ancho, height = alto,
      family = "serif")
}

paleta_heat <- colorRampPalette(c("white", "#74c476", "#006d2c"))(100)

for (tipo_g in c("IND", "DEP")) {
  for (bloque_g in c("n>p", "p>n")) {
    for (config_g in c("I", "II")) {
      for (n_g in sort(unique(
        subset(escenarios, tipo == tipo_g & bloque == bloque_g &
               config == config_g)$n))) {
        
        sub_g <- subset(resultados_df,
                        tipo == tipo_g & bloque == bloque_g &
                          config == config_g & n == n_g)
        if (nrow(sub_g) == 0) next
        
        p_g     <- unique(sub_g$p)
        mets_g  <- ORDEN_METODOS[ORDEN_METODOS %in% sub_g$Metodo]
        sub_g   <- sub_g[match(mets_g, sub_g$Metodo), ]
        n_met_g <- length(mets_g)
        
        # Matriz [n_met x p_g]: todas las covariables
        tpr_cols <- paste0("TPR_X", 1:p_g)
        mat_heat <- as.matrix(sub_g[, tpr_cols])
        rownames(mat_heat) <- mets_g
        colnames(mat_heat) <- paste0("X", 1:p_g)
        
        fname_h <- sprintf("heatmap_%s_%s_cfg%s_n%d.pdf",
                           tolower(tipo_g), gsub(">", "", bloque_g),
                           config_g, n_g)
        
        # Ancho proporcional al número de covariables
        ancho_h <- max(8, 4 + p_g * 0.12)
        alto_h  <- max(3.5, n_met_g * 0.65 + 2)
        
        pdf(file.path(DIR_OUT, fname_h),
            width = ancho_h, height = alto_h, family = "serif")
        
        # Layout: panel principal + barra de color (termómetro)
        layout(matrix(c(1, 2), nrow = 1), widths = c(10, 1))
        
        # --- Panel principal: heatmap ---
        par(mar = c(6, 7, 5, 1))
        
        image(x    = 1:p_g,
              y    = 1:n_met_g,
              z    = t(mat_heat),
              col  = paleta_heat,
              zlim = c(0, 1),
              axes = FALSE,
              xlab = "",
              ylab = "")
        
        title(main = sprintf("Proporción de selección - %s | Config. %s | n=%d | p=%d",
                             paste0(tipo_g, " ", config_g), config_g, n_g, p_g),
              cex.main = 1.5)
        
        # Eje X: todas las covariables, etiquetas más pequeñas si p es grande
        cex_x <- if (p_g > 60) 0.45 else if (p_g > 20) 0.6 else 0.8
        axis(1,
             at     = 1:p_g,
             labels = paste0("X", 1:p_g),
             las    = 2, cex.axis = cex_x)
        mtext("Covariable", side = 1, line = 4.5, cex = 0.85)
        
        # Eje Y: métodos en orden canónico (Forward arriba)
        axis(2,
             at     = 1:n_met_g,
             labels = mets_g,
             las    = 2, cex.axis = 1.1)
        
        # Línea vertical separando relevantes (X1-X5) de ruido (X6...Xp)
        abline(v = N_REL + 0.5, col = "black", lwd = 1.5)
        
        box()
        
        # --- Panel lateral: termómetro de color ---
        par(mar = c(6, 1, 5, 3))
        leyenda_z <- matrix(seq(0, 1, length.out = 100), nrow = 1)
        image(x = 1, y = seq(0, 1, length.out = 100), z = leyenda_z,
              col = paleta_heat, axes = FALSE, xlab = "", ylab = "")
        axis(4, at = seq(0, 1, by = 0.25), las = 2, cex.axis = 0.75)
        mtext("Prop.", side = 3, line = 0.5, cex = 0.7)
        box()
        
        dev.off()
        cat(sprintf(">> Heatmap: %s\n", fname_h))
      }
    }
  }
}


# ============================================================================ #
# 7b. HEATMAPS MAESTROS POR ESCENARIO (paneles por n) ------------------------
# ============================================================================ #
# Un heatmap por tipo x config x bloque, con un panel por cada n,
# todos compartiendo la misma escala de color.

for (tipo_g in c("IND", "DEP")) {
  for (config_g in c("I", "II")) {
    for (bloque_g in c("n>p", "p>n")) {
      
      sub_g <- subset(resultados_df,
                      tipo == tipo_g & config == config_g & bloque == bloque_g)
      if (nrow(sub_g) == 0) next
      
      n_vals  <- sort(unique(sub_g$n))
      p_g     <- unique(sub_g$p)
      mets_g  <- ORDEN_METODOS[ORDEN_METODOS %in% sub_g$Metodo]
      n_met_g <- length(mets_g)
      
      fname_hm <- sprintf("heatmap_maestro_%s_%s_cfg%s.pdf",
                          tolower(tipo_g), gsub(">", "", bloque_g), config_g)
      
      ancho_h <- max(10, 3 + p_g * 0.10) * length(n_vals) * 0.6
      alto_h  <- max(3.5, n_met_g * 0.65 + 2.5)
      
      pdf(file.path(DIR_OUT, fname_hm),
          width = ancho_h, height = alto_h, family = "serif")
      
      layout(matrix(c(seq_along(n_vals), length(n_vals) + 1), nrow = 1),
             widths = c(rep(10, length(n_vals)), 1.2))
      
      cex_x <- if (p_g > 60) 0.45 else if (p_g > 20) 0.6 else 0.8
      
      for (n_g in n_vals) {
        sub_n <- subset(sub_g, n == n_g)
        sub_n <- sub_n[match(mets_g, sub_n$Metodo), ]
        
        tpr_cols <- paste0("TPR_X", 1:p_g)
        mat_heat <- as.matrix(sub_n[, tpr_cols])
        
        par(mar = c(6, 7, 4, 1))
        image(x = 1:p_g, y = 1:n_met_g, z = t(mat_heat),
              col = paleta_heat, zlim = c(0, 1),
              axes = FALSE, xlab = "", ylab = "")
        title(main = sprintf("n=%d", n_g), cex.main = 1.0)
        axis(1, at = 1:p_g, labels = paste0("X", 1:p_g),
             las = 2, cex.axis = cex_x)
        axis(2, at = 1:n_met_g, labels = mets_g, las = 2, cex.axis = 0.85)
        abline(v = N_REL + 0.5, col = "black", lwd = 1.5)
        box()
      }
      
      # Termómetro común
      par(mar = c(6, 1, 4, 3))
      leyenda_z <- matrix(seq(0, 1, length.out = 100), nrow = 1)
      image(x = 1, y = seq(0, 1, length.out = 100), z = leyenda_z,
            col = paleta_heat, axes = FALSE, xlab = "", ylab = "")
      axis(4, at = seq(0, 1, by = 0.25), las = 2, cex.axis = 0.8)
      mtext("Prop.", side = 3, line = 0.5, cex = 0.8)
      box()
      
      mtext(sprintf("%s | %s | p = %d", paste0(tipo_g, " ", config_g), bloque_g, p_g),
            outer = TRUE, line = -1.5, cex = 1.4, font = 2)
      
      dev.off()
      cat(sprintf(">> Heatmap maestro: %s\n", fname_hm))
    }
  }
}

# ============================================================================ #
# 8. GUARDAR RESULTADOS FINALES ------
# ============================================================================ #

saveRDS(resultados_df, file.path(DIR_OUT, "resultados_finales.rds"))

cat("\n=============================================================\n")
cat("ARCHIVOS GENERADOS EN:", DIR_OUT, "\n\n")
cat("  Fase 1:   betas_escXX_FINAL.rds  (betas + mse por escenario)\n")
cat("  Fase 2:   resultados_finales.rds\n")
cat("  Tablas:   tabla_<tipo>_<bloque>_p*_n*_cfg*.tex  (individuales, apéndice)\n")
cat("            tabla_unif_<tipo>_<bloque>_cfg*.tex   (unificadas)\n")
cat("  Heatmaps: heatmap_<tipo>_<bloque>_cfg*_n*.pdf  (todas las covariables)\n")
cat("=============================================================\n")