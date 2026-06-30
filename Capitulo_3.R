# =============================================================================
# CAPÍTULO 3 – MÉTODOS DE SELECCIÓN DE VARIABLES
# Script R completo para generar todas las tablas y figuras ilustrativas
# TFG: Procesos de selección de covariables en el modelo de regresión lineal
# =============================================================================
# Paquetes necesarios.
#   install.packages(c("glmnet","ncvreg","leaps","mboost","xtable"))
# =============================================================================

library(glmnet)    # LASSO y Adaptive LASSO
library(ncvreg)    # SCAD
library(leaps)     # Best subset selection
library(mboost)    # Boosting lineal (L2-boosting)
library(xtable)    # Exportar tablas a LaTeX

# Directorio de salida para las figuras 
DIR_FIGS <- "Cap3"
if (!dir.exists(DIR_FIGS)) dir.create(DIR_FIGS, recursive = TRUE)

# Función auxiliar para guardar figuras en PDF
guardar_pdf <- function(nombre, ancho = 7, alto = 5) {
  pdf(file.path(DIR_FIGS, nombre), width = ancho, height = alto,
      family = "serif")
}

# =============================================================================
# 0. MODELO SIMULADO – HILO CONDUCTOR DE TODO EL CAPÍTULO 3
# =============================================================================
# Modelo verdadero:
#   Y = 4 + 3*X1 - 2*X3 + 1.5*X7 + epsilon
#
# Covariables: X1 ... X10
#   - X1, X3, X7: efectos reales (beta = 3, -2, 1.5)
#   - X2: correlacionada con X3 (rho = 0.85) → ruido correlacionado
#   - X4, X5, X6, X8, X9, X10: ruido puro (beta = 0)
#
# Diseño: n = 200, p = 10, sigma^2 = 4 (sd = 2)
# Semilla fija para reproducibilidad
# =============================================================================

set.seed(2024)
n  <- 200   # tamaño muestral
p  <- 10    # número de covariables

# ── Estructura de correlación ──────────────────────────────────────────────
# X2 correlacionada con X3; el resto independientes
Sigma <- diag(p)
Sigma[2, 3] <- Sigma[3, 2] <- 0.85   # cor(X2, X3) = 0.85

L <- chol(Sigma)
Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
X <- Z %*% L
colnames(X) <- paste0("X", 1:p)

# ── Variable respuesta ─────────────────────────────────────────────────────
beta_verdadero <- c(3, 0, -2, 0, 0, 0, 1.5, 0, 0, 0)   # X1, X3, X7
intercepto     <- 4
epsilon        <- rnorm(n, mean = 0, sd = 2)             # sigma^2 = 4
Y <- intercepto + X %*% beta_verdadero + epsilon

# Data frame completo
datos <- as.data.frame(cbind(Y = as.numeric(Y), X))

cat("=== Resumen del modelo simulado ===\n")
cat("n =", n, "| p =", p, "\n")
cat("Coeficientes verdaderos:\n")
print(data.frame(Variable = paste0("X", 1:p),
                 Beta_verdadero = beta_verdadero))
cat("Variables relevantes: X1 (3), X3 (-2), X7 (1.5)\n")
cat("Variables ruido: X2(*), X4, X5, X6, X8, X9, X10\n")
cat("(*) X2 correlacionada con X3 (rho = 0.85)\n\n")

# =============================================================================
# 3.1  MÉTODOS BASADOS EN INFERENCIA CLÁSICA
# =============================================================================

cat("=== SECCIÓN 3.1: Contrastes t en el modelo completo ===\n")

modelo_completo <- lm(Y ~ ., data = datos)
resumen         <- summary(modelo_completo)

# ── Tabla: Coeficientes estimados con p-valores ───────────────────────────
coef_tabla <- as.data.frame(resumen$coefficients)
coef_tabla <- coef_tabla[-1, ]   # quitar intercepto
colnames(coef_tabla) <- c("Estimación", "Error estándar", "Estadístico t", "p-valor")
coef_tabla$Significativo <- ifelse(coef_tabla[["p-valor"]] < 0.05, "Sí", "No")

cat("\nTabla de coeficientes (modelo completo con 10 covariables):\n")
coef_tabla_imp <- coef_tabla
coef_tabla_imp[sapply(coef_tabla_imp, is.numeric)] <-
  round(coef_tabla_imp[sapply(coef_tabla_imp, is.numeric)], 4)
print(coef_tabla_imp)

# Exportar a LaTeX
coef_tabla_tex <- coef_tabla
coef_tabla_tex[sapply(coef_tabla_tex, is.numeric)] <-
  round(coef_tabla_tex[sapply(coef_tabla_tex, is.numeric)], 4)

# Variables relevantes para colorear
vars_relevantes <- c("X1", "X3", "X7")

# Colorear fila entera para las variables relevantes
coef_tabla_color <- coef_tabla_tex
rownames_col <- rownames(coef_tabla_color)

rownames_col <- ifelse(rownames_col %in% vars_relevantes,
                       paste0("\\rowcolor{green!25}", rownames_col),
                       rownames_col)
rownames(coef_tabla_color) <- rownames_col

tabla_tex <- xtable(
  coef_tabla_color,
  caption = "Estimaciones OLS del modelo completo y resultado de los contrastes $t$ individuales ($\\alpha = 0.05$). Las covariables con efecto verdadero distinto de cero son $X_1$, $X_3$ y $X_7$.",
  label   = "tab:C3:contrastes_t",
  digits  = c(0, 3, 3, 3, 4, 0)
)
sink(file.path(DIR_FIGS, "tabla_contrastes_t.tex"))
print(tabla_tex,
      include.rownames       = TRUE,
      booktabs               = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_contrastes_t.tex\n\n")

# ── Figura: Gráfico de p-valores (-log10) ────────────────────────────────
guardar_pdf("fig_pvalores_t.pdf", ancho = 7, alto = 4.5)
pvals   <- coef_tabla[["p-valor"]]
nombres <- rownames(coef_tabla)
colores <- ifelse(pvals < 0.05, "#2166ac", "#d73027")

par(mar = c(4, 5, 3.5, 2), family = "serif")
barplot(
  -log10(pvals),
  names.arg = nombres,
  col       = colores,
  las       = 2,
  ylab      = expression(-log[10](p-valor)),
  main      = "Contrastes t individuales — modelo completo",
  cex.names = 0.85,
  cex.axis  = 0.85,
  border    = "white"
)
abline(h = -log10(0.05), lty = 2, col = "black", lwd = 1.5)
legend("topright",
       legend = c("p < 0.05 (significativo)", "p >= 0.05 (no significativo)"),
       fill   = c("#2166ac", "#d73027"),
       border = "white", bty = "n", cex = 0.8)
dev.off()
cat(">> Guardada: fig_pvalores_t.pdf\n\n")


# ── Contraste F parcial (ejemplo: bloque X2, X4, X5, X6, X8, X9, X10) ───
# Se compara el modelo completo con el modelo reducido (solo X1, X3, X7)
cat("=== SECCIÓN 3.1: Contraste F parcial (modelo reducido vs completo) ===\n")

modelo_reducido <- lm(Y ~ X1 + X3 + X7, data = datos)

# anova() calcula el contraste F entre modelos anidados
resultado_F <- anova(modelo_reducido, modelo_completo)
cat("\nContraste F — modelo reducido {X1,X3,X7} vs modelo completo:\n")
print(resultado_F)

SQR_R <- sum(resid(modelo_reducido)^2)
SQR_C <- sum(resid(modelo_completo)^2)
q     <- 7   # variables eliminadas: X2, X4, X5, X6, X8, X9, X10
F_man <- ((SQR_R - SQR_C) / q) / (SQR_C / (n - p - 1))
pval_F <- pf(F_man, df1 = q, df2 = n - p - 1, lower.tail = FALSE)

cat(sprintf("\nCálculo manual: F = %.4f | p-valor = %.4f\n", F_man, pval_F))
cat("Conclusión: no se rechaza H0, las 7 variables ruido no aportan conjuntamente.\n\n")

# Tabla LaTeX del contraste F
tabla_F <- data.frame(
  Modelo     = c("Reducido ($X_1, X_3, X_7$)", "Completo (10 covariables)"),
  "GL res."  = c(n - 3 - 1, n - p - 1),
  "SQR"      = round(c(SQR_R, SQR_C), 3),
  "Estadístico F" = c(NA, round(F_man, 4)),
  "p-valor"  = c(NA, round(pval_F, 4)),
  check.names = FALSE
)
tabla_F_tex <- xtable(
  tabla_F,
  caption = "Contraste $F$ parcial comparando el modelo reducido $\\{X_1, X_3, X_7\\}$ con el modelo completo. El p-valor elevado indica que las siete covariables eliminadas no aportan conjuntamente información significativa.",
  label   = "tab:C3:contraste_F",
  digits  = c(0, 0, 0, 3, 4, 4)
)
sink(file.path(DIR_FIGS, "tabla_contraste_F.tex"))
print(tabla_F_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_contraste_F.tex\n\n")


# =============================================================================
# 3.2  MÉTODOS CLÁSICOS DE BÚSQUEDA DE MODELOS
# =============================================================================

# ── 3.2.1  Best Subset Selection ─────────────────────────────────────────
cat("=== SECCIÓN 3.2.1: Best Subset Selection ===\n")

best_sub    <- regsubsets(Y ~ ., data = datos, nvmax = p, method = "exhaustive")
resumen_bs  <- summary(best_sub)

bic_vals  <- resumen_bs$bic
radj_vals <- resumen_bs$adjr2
tamano    <- 1:p

mejor_bic  <- which.min(bic_vals)
mejor_radj <- which.max(radj_vals)

cat("\nMejor modelo según BIC: tamaño =", mejor_bic, "\n")
cat("Variables incluidas:\n")
print(names(which(resumen_bs$which[mejor_bic, -1])))

cat("\nBIC por tamaño:\n")
print(round(data.frame(Tamaño = tamano, BIC = bic_vals, R2adj = radj_vals), 3))

# Tabla LaTeX: evolución BIC y R² ajustado
bs_tabla <- data.frame(
  "Tamaño" = tamano,
  "BIC"    = round(bic_vals, 2),
  "$R^2_{\\text{aj}}$" = round(radj_vals, 4),
  "Mejor (BIC)" = ifelse(tamano == mejor_bic, "\\checkmark", ""),
  check.names = FALSE
)
tabla_bs_tex <- xtable(
  bs_tabla,
  caption = "Evolución del BIC y del $R^2$ ajustado en \\textit{best subset selection}. El modelo óptimo según BIC (tamaño 3) se marca con \\checkmark.",
  label   = "tab:C3:best_subset",
  digits  = c(0, 0, 2, 4, 0)
)
sink(file.path(DIR_FIGS, "tabla_best_subset.tex"))
print(tabla_bs_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_best_subset.tex\n")

# ── Figura: Best Subset — BIC y R² ajustado ───────────────────────────────
# Se usan dos paneles separados con márgenes amplios para evitar solapamiento
guardar_pdf("fig_best_subset.pdf", ancho = 9, alto = 4.5)
par(mfrow = c(1, 2), mar = c(5, 4.5, 4, 2), family = "serif")

# Panel izquierdo: BIC
plot(tamano, bic_vals,
     type = "b", pch = 19, col = "#2166ac",
     xlab = "Número de covariables",
     ylab = "BIC",
     main = "Best subset — BIC por tamaño",
     cex  = 0.9, cex.main = 0.95, cex.lab = 0.9)
points(mejor_bic, bic_vals[mejor_bic],
       pch = 19, col = "#d73027", cex = 2)
legend("topright",
       legend = paste0("Mínimo BIC\n(k = ", mejor_bic, ")"),
       pch = 19, col = "#d73027",
       bty = "n", cex = 0.8)

# Panel derecho: R² ajustado
plot(tamano, radj_vals,
     type = "b", pch = 19, col = "#4dac26",
     xlab = "Número de covariables",
     ylab = expression(R^2~ajustado),
     main = expression(paste("Best subset — ", R^2, " ajustado")),
     cex  = 0.9, cex.main = 0.95, cex.lab = 0.9)
points(mejor_radj, radj_vals[mejor_radj],
       pch = 19, col = "#d73027", cex = 2)
legend("bottomright",
       legend = paste0("Máximo ", expression(R^2), " aj.\n(k = ", mejor_radj, ")"),
       pch = 19, col = "#d73027",
       bty = "n", cex = 0.8)
dev.off()
cat(">> Guardada: fig_best_subset.pdf\n\n")

# ── Figura adicional: variables seleccionadas en cada tamaño (heatmap) ────
guardar_pdf("fig_best_subset_heatmap.pdf", ancho = 9, alto = 5)
par(mar = c(5, 5, 4, 2), family = "serif")

# Matriz booleana: filas = tamaños, columnas = covariables (sin intercepto)
which_mat <- resumen_bs$which[, -1]
which_mat_inv <- which_mat[nrow(which_mat):1, ]

image(t(which_mat_inv),
      col  = c("white", "#2166ac"),
      axes = FALSE,
      main = "Best subset — variables incluidas por tamaño de modelo",
      cex.main = 1.1)

# Líneas de cuadrícula para separar celdas
abline(v = seq(-1/(2*(p-1)), 1 + 1/(2*(p-1)), length.out = p + 1),
       col = "grey70", lwd = 0.5)
abline(h = seq(-1/(2*(p-1)), 1 + 1/(2*(p-1)), length.out = p + 1),
       col = "grey70", lwd = 0.5)

axis(1,
     at     = seq(0, 1, length.out = p),
     labels = paste0("X", 1:p),
     las    = 1, cex.axis = 0.85)
axis(2,
     at     = seq(0, 1, length.out = p),
     labels = paste0("k = ", p:1),   # invertido: k=10 arriba, k=1 abajo
     las    = 2, cex.axis = 0.85)

# Línea del modelo óptimo por BIC (ajustada al eje invertido)
abline(h = (p - mejor_bic) / (p - 1), lty = 2, col = "#d73027", lwd = 1.5)

legend("topright",
       legend = c("Incluida", paste0("Óptimo BIC (k=", mejor_bic, ")")),
       fill   = c("#2166ac", NA),
       lty    = c(NA, 2),
       col    = c(NA, "#d73027"),
       border = c("grey70", NA),
       bg     = "white",          
       box.col = "grey70",        
       bty    = "o", cex = 0.85)
dev.off()
cat(">> Guardada: fig_best_subset_heatmap.pdf\n\n")


# ── 3.2.2 / 3.2.3  Forward, Backward y Stepwise ──────────────────────────
cat("=== SECCIÓN 3.2.2-3: Forward / Backward / Stepwise ===\n")

# regsubsets para trayectorias BIC paso a paso
fwd     <- regsubsets(Y ~ ., data = datos, nvmax = p, method = "forward")
res_fwd <- summary(fwd)
bwd     <- regsubsets(Y ~ ., data = datos, nvmax = p, method = "backward")
res_bwd <- summary(bwd)

# step() sobre lm — utiliza AIC por defecto
# trace = 1 para capturar los pasos; redirigimos a texto para inspeccionarlos
modelo_nulo     <- lm(Y ~ 1, data = datos)
modelo_step_fw  <- step(modelo_nulo,
                        scope     = list(lower = ~1,
                                         upper = formula(modelo_completo)),
                        direction = "forward",  trace = 0)
modelo_step_bw  <- step(modelo_completo,
                        direction = "backward", trace = 0)
modelo_step_bth <- step(modelo_completo,
                        direction = "both",     trace = 0)

cat("Forward  AIC:", round(AIC(modelo_step_fw),  2),
    "| Variables:", paste(names(coef(modelo_step_fw))[-1],  collapse = ", "), "\n")
cat("Backward AIC:", round(AIC(modelo_step_bw),  2),
    "| Variables:", paste(names(coef(modelo_step_bw))[-1],  collapse = ", "), "\n")
cat("Stepwise AIC:", round(AIC(modelo_step_bth), 2),
    "| Variables:", paste(names(coef(modelo_step_bth))[-1], collapse = ", "), "\n\n")

# ── Tabla numérica: BIC y variables por paso — Forward ────────────────────
# Para cada tamaño k, extraemos la variable incorporada en ese paso
vars_fwd <- character(p)
for (k in 1:p) {
  incluidas <- names(which(res_fwd$which[k, -1]))   # sin intercepto
  if (k == 1) {
    vars_fwd[k] <- incluidas
  } else {
    anteriores  <- names(which(res_fwd$which[k - 1, -1]))
    nueva       <- setdiff(incluidas, anteriores)
    vars_fwd[k] <- ifelse(length(nueva) > 0, nueva, "—")
  }
}

tabla_fwd <- data.frame(
  Paso                  = 1:p,
  "Variable incorporada" = vars_fwd,
  BIC                   = round(res_fwd$bic, 2),
  "$R^2_{\\text{aj}}$"  = round(res_fwd$adjr2, 4),
  check.names = FALSE
)

cat("\nTabla Forward selection — BIC y R² ajustado por paso:\n")
print(tabla_fwd)

# Colorear fila de variable incorporada cuando es relevante
pasos_color <- ifelse(vars_fwd %in% vars_relevantes,
                      paste0("\\rowcolor{green!25}", 1:p),
                      as.character(1:p))
vars_fwd_color <- vars_fwd   # sin color en la celda, lo lleva la fila

tabla_fwd_color <- data.frame(
  Paso                   = pasos_color,
  "Variable incorporada" = vars_fwd_color,
  BIC                    = round(res_fwd$bic, 2),
  "$R^2_{\\text{aj}}$"   = round(res_fwd$adjr2, 4),
  check.names = FALSE
)

tabla_fwd_tex <- xtable(
  tabla_fwd_color,
  caption = "Evolución del BIC y del $R^2$ ajustado en cada paso de forward selection. En cada paso se incorpora la covariable que produce la mayor mejora. El mínimo BIC se alcanza en el paso~3, con el modelo $\\{X_1, X_3, X_7\\}$.",
  label   = "tab:C3:forward_bic",
  digits  = c(0, 0, 0, 2, 4)
)
sink(file.path(DIR_FIGS, "tabla_forward_bic.tex"))
print(tabla_fwd_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_forward_bic.tex\n")

# ── Tabla numérica: BIC y variables por paso — Backward ───────────────────
# En backward, el paso k corresponde a un modelo con (p - k + 1) variables:
# en paso 1 tenemos p variables, en paso p tenemos 1 variable.
# regsubsets ordena por tamaño creciente, así que invertimos para la tabla.
bic_bwd_ord  <- rev(res_bwd$bic)
radj_bwd_ord <- rev(res_bwd$adjr2)

vars_bwd <- character(p)
vars_bwd[1] <- "— (modelo completo)"
for (k in 2:p) {
  tam_prev <- p - k + 2   # tamaño del modelo en el paso anterior
  tam_curr <- p - k + 1   # tamaño del modelo en este paso
  incl_prev <- names(which(res_bwd$which[tam_prev, -1]))
  incl_curr <- names(which(res_bwd$which[tam_curr, -1]))
  eliminada <- setdiff(incl_prev, incl_curr)
  vars_bwd[k] <- ifelse(length(eliminada) > 0, eliminada[1], "—")
}

cat("\nVariables eliminadas en backward:\n")
print(vars_bwd)


resaltar_relevantes <- function(cadena, vars_rel) {
  for (v in vars_rel) {
    cadena <- gsub(
      paste0("(^|,\\s*)(", v, ")(\\s*,|$)"),
      paste0("\\1\\\\colorbox{green!25}{\\2}\\3"),
      cadena,
      perl = TRUE
    )
  }
  cadena
}


vars_bwd_color <- vars_bwd


vars_en_modelo_raw <- sapply(p:1, function(k)
  paste(names(which(res_bwd$which[k, -1])), collapse = ", "))

vars_en_modelo_color <- sapply(vars_en_modelo_raw,
                               resaltar_relevantes,
                               vars_rel = vars_relevantes)

tabla_bwd_color <- data.frame(
  Paso                  = 1:p,
  "Variable eliminada"  = vars_bwd_color,
  "Variables en modelo" = vars_en_modelo_color,
  BIC                   = round(bic_bwd_ord, 2),
  "$R^2_{\\text{aj}}$"  = round(radj_bwd_ord, 4),
  check.names = FALSE
)


tabla_bwd_tex <- xtable(
  tabla_bwd_color,
  caption = "Evolución del BIC y del $R^2$ ajustado en cada paso de backward elimination. En cada paso se elimina la covariable menos relevante. El mínimo BIC se alcanza cuando el modelo queda reducido a tres covariables. En verde se resaltan las variables relevantes $X_1$, $X_3$ y $X_7$ cuando están presentes en el modelo.",
  label   = "tab:C3:backward_bic",
  digits  = c(0, 0, 0, 0, 2, 4)
)
sink(file.path(DIR_FIGS, "tabla_backward_bic.tex"))
print(tabla_bwd_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_backward_bic.tex\n")

# ── Stepwise con BIC usando regsubsets(method="seqrep") ───────────────────
# regsubsets con method="seqrep" implementa búsqueda secuencial con revisión
# en ambas direcciones (equivalente a stepwise). Usamos BIC como criterio,
# coherente con las tablas de forward y backward.
stp     <- regsubsets(Y ~ ., data = datos, nvmax = p, method = "seqrep")
res_stp <- summary(stp)

mejor_stp <- which.min(res_stp$bic)
cat("\nStepwise (BIC) — mejor tamaño:", mejor_stp, "\n")
cat("Variables:", paste(names(which(res_stp$which[mejor_stp, -1])), collapse = ", "), "\n\n")

# Reconstruir trayectoria: qué variable entra o sale en cada paso
# Comparamos el modelo de tamaño k con el de tamaño k-1 para detectar cambios
accion_stp <- character(p)
vars_stp   <- character(p)
for (k in 1:p) {
  curr <- names(which(res_stp$which[k, -1]))
  vars_stp[k] <- paste(curr, collapse = ", ")
  if (k == 1) {
    accion_stp[k] <- paste0("+ ", curr)
  } else {
    prev     <- names(which(res_stp$which[k - 1, -1]))
    entran   <- setdiff(curr, prev)
    salen    <- setdiff(prev, curr)
    accion_k <- c(if (length(entran) > 0) paste0("+ ", entran),
                  if (length(salen)  > 0) paste0("− ", salen))
    accion_stp[k] <- paste(accion_k, collapse = " / ")
  }
}

tabla_stp <- data.frame(
  Paso               = 1:p,
  "Acción"           = accion_stp,
  "Modelo actual"    = vars_stp,
  BIC                = round(res_stp$bic, 2),
  "$R^2_{\\text{aj}}$" = round(res_stp$adjr2, 4),
  check.names = FALSE
)

cat("Tabla Stepwise — evolución BIC:\n")
print(tabla_stp)

# Normalizar el signo menos Unicode en la columna Acción (sin colores)
accion_stp_limpia <- sapply(accion_stp, function(acc) {
  gsub("\u2212", "$-$", acc)
})

# Colorbox verde en X1, X3, X7 dentro de "Modelo actual"
vars_stp_color <- sapply(vars_stp,
                         resaltar_relevantes,
                         vars_rel = vars_relevantes)

tabla_stp_color <- data.frame(
  Paso                 = 1:p,
  "Acción"             = accion_stp_limpia,
  "Modelo actual"      = vars_stp_color,
  BIC                  = round(res_stp$bic, 2),
  "$R^2_{\\text{aj}}$" = round(res_stp$adjr2, 4),
  check.names = FALSE
)

tabla_stp_tex <- xtable(
  tabla_stp_color,
  caption = "Evolución del BIC en el procedimiento stepwise (búsqueda en ambas direcciones). Cada paso indica si se incorpora ($+$) o elimina ($-$) una covariable. En verde se resaltan las variables relevantes $X_1$, $X_3$ y $X_7$ cuando están presentes en el modelo.",
  label   = "tab:C3:stepwise_bic",
  digits  = c(0, 0, 0, 0, 2, 4)
)
sink(file.path(DIR_FIGS, "tabla_stepwise_bic.tex"))
print(tabla_stp_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_stepwise_bic.tex\n")
# ── Figura: Trayectorias BIC — Forward y Backward ────────────────────────
guardar_pdf("fig_forward_trayectoria.pdf", ancho = 9, alto = 4.5)
par(mfrow = c(1, 2), mar = c(5, 4.5, 4, 2), family = "serif")

mejor_fwd <- which.min(res_fwd$bic)
plot(1:p, res_fwd$bic,
     type = "b", pch = 19, col = "#2166ac",
     xlab = "Covariables incorporadas", ylab = "BIC",
     main = "Forward selection — BIC",
     cex  = 0.9, cex.main = 0.95, cex.lab = 0.9)
points(mejor_fwd, res_fwd$bic[mejor_fwd],
       pch = 19, col = "#d73027", cex = 2)
legend("topright",
       legend = paste0("Mínimo BIC (k=", mejor_fwd, ")"),
       pch = 19, col = "#d73027", bty = "n", cex = 0.82)

mejor_bwd <- which.min(res_bwd$bic)
plot(1:p, res_bwd$bic,
     type = "b", pch = 19, col = "#4dac26",
     xlab = "Covariables en el modelo", ylab = "BIC",
     main = "Backward elimination — BIC",
     cex  = 0.9, cex.main = 0.95, cex.lab = 0.9)
points(mejor_bwd, res_bwd$bic[mejor_bwd],
       pch = 19, col = "#d73027", cex = 2)
legend("topright",
       legend = paste0("Mínimo BIC (k=", mejor_bwd, ")"),
       pch = 19, col = "#d73027", bty = "n", cex = 0.82)
dev.off()
cat(">> Guardada: fig_forward_trayectoria.pdf\n\n")


# =============================================================================
# 3.3  MÉTODOS DE REGULARIZACIÓN: LASSO, ADAPTIVE LASSO, SCAD
# =============================================================================

X_mat <- as.matrix(datos[, -1])
Y_vec <- datos$Y

# ── 3.3.1  LASSO ─────────────────────────────────────────────────────────
cat("=== SECCIÓN 3.3.1: LASSO ===\n")

lasso_fit  <- glmnet(X_mat, Y_vec, alpha = 1, standardize = TRUE)
lasso_cv   <- cv.glmnet(X_mat, Y_vec, alpha = 1, nfolds = 10,
                        standardize = TRUE)
lambda_opt <- lasso_cv$lambda.min

cat("Lambda óptimo (CV 10-fold):", round(lambda_opt, 5), "\n")

coef_lasso <- as.matrix(coef(lasso_fit, s = lambda_opt))[-1, ]
cat("Coeficientes LASSO en lambda óptimo:\n")
print(round(coef_lasso, 4))


cat("Lambda min (LASSO):", round(lasso_cv$lambda.min, 5), "\n")
cat("Lambda 1se (LASSO):", round(lasso_cv$lambda.1se, 5), "\n")
cat("Variables activas en lambda.min:", sum(round(coef_lasso, 4) != 0), "\n")
cat("Variables activas en lambda.1se:",
    sum(round(as.numeric(coef(lasso_fit, s = lasso_cv$lambda.1se))[-1], 4) != 0), "\n")

color_latex <- c(
  "\\cellcolor[HTML]{E41A1C}",   # X1  rojo
  "\\cellcolor[HTML]{AAAAAA}",   # X2  gris
  "\\cellcolor[HTML]{377EB8}",   # X3  azul
  "\\cellcolor[HTML]{BBBBBB}",   # X4  gris claro
  "\\cellcolor[HTML]{CCCCCC}",   # X5
  "\\cellcolor[HTML]{DDDDDD}",   # X6
  "\\cellcolor[HTML]{4DAF4A}",   # X7  verde
  "\\cellcolor[HTML]{BBBBBB}",   # X8
  "\\cellcolor[HTML]{CCCCCC}",   # X9
  "\\cellcolor[HTML]{DDDDDD}"    # X10
)

# Tabla LASSO con columna de color
tabla_lasso <- data.frame(
  Color          = color_latex,
  Variable       = paste0("X", 1:p),
  Beta_verdadero = beta_verdadero,
  LASSO          = round(coef_lasso, 4),
  Seleccionado   = ifelse(round(coef_lasso, 4) != 0, "Sí", "No"),
  check.names = FALSE
)
print(tabla_lasso)

tabla_lasso_tex <- xtable(
  tabla_lasso,
  caption = "Coeficientes estimados por LASSO en $\\hat{\\lambda}$ (validación cruzada 10-fold) comparados con los valores verdaderos. La columna de color identifica cada variable con el color de su curva en la Figura~\\ref{fig:lasso_path}.",
  label   = "tab:C3:lasso",
  digits  = c(0, 0, 0, 0, 4, 0)
)
sink(file.path(DIR_FIGS, "tabla_lasso.tex"))
print(tabla_lasso_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_lasso.tex\n\n")

# ── 3.3.2  Adaptive LASSO ────────────────────────────────────────────────
cat("=== SECCIÓN 3.3.2: Adaptive LASSO ===\n")

# Paso 1: estimación inicial con Ridge
ridge_init <- cv.glmnet(X_mat, Y_vec, alpha = 0, standardize = TRUE)
beta_ridge  <- as.numeric(coef(ridge_init, s = ridge_init$lambda.min))[-1]

# Pesos adaptativos: w_j = 1 / |beta_ridge_j|^gamma  (gamma = 1)
gamma_alasso <- 1
pesos_adapt  <- 1 / (abs(beta_ridge) + 1e-6)^gamma_alasso

cat("Pesos adaptativos (mayores = más penalizados):\n")
print(round(setNames(pesos_adapt, paste0("X", 1:p)), 4))

# Paso 2: LASSO ponderado
alasso_fit <- glmnet(X_mat, Y_vec, alpha = 1,
                     penalty.factor = pesos_adapt, standardize = TRUE)
alasso_cv  <- cv.glmnet(X_mat, Y_vec, alpha = 1,
                        penalty.factor = pesos_adapt, nfolds = 10,
                        standardize = TRUE)
lambda_al  <- alasso_cv$lambda.min

cat("Lambda óptimo Adaptive LASSO (CV 10-fold):", round(lambda_al, 5), "\n")

coef_alasso <- as.numeric(coef(alasso_fit, s = lambda_al))[-1]
names(coef_alasso) <- paste0("X", 1:p)
cat("Coeficientes Adaptive LASSO:\n")
print(round(coef_alasso, 4))


cat("Lambda min (Adaptive LASSO):", round(alasso_cv$lambda.min, 5), "\n")
cat("Lambda 1se (Adaptive LASSO):", round(alasso_cv$lambda.1se, 5), "\n")
cat("Variables activas en lambda.min:", sum(round(coef_alasso, 4) != 0), "\n")
cat("Variables activas en lambda.1se:",
    sum(round(as.numeric(coef(alasso_fit, s = alasso_cv$lambda.1se))[-1], 4) != 0), "\n")

# Tabla Adaptive LASSO (mismo formato que tabla LASSO)
tabla_alasso <- data.frame(
  Variable         = paste0("X", 1:p),
  "Beta Verdadero" = beta_verdadero,
  "Adaptive LASSO" = round(coef_alasso, 4),
  Seleccionado     = ifelse(round(coef_alasso, 4) != 0, "Sí", "No"),
  check.names = FALSE
)
print(tabla_alasso)

tabla_alasso_tex <- xtable(
  tabla_alasso,
  caption = "Coeficientes estimados por Adaptive LASSO en $\\hat{\\lambda}$ (validación cruzada 10-fold) comparados con los valores verdaderos.",
  label   = "tab:C3:alasso",
  digits  = c(0, 0, 0, 4, 0)
)
sink(file.path(DIR_FIGS, "tabla_alasso.tex"))
print(tabla_alasso_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_alasso.tex\n\n")

# ── 3.3.3  SCAD ──────────────────────────────────────────────────────────
cat("=== SECCIÓN 3.3.3: SCAD ===\n")

scad_cv     <- cv.ncvreg(X_mat, Y_vec, penalty = "SCAD", nfolds = 10)
lambda_scad <- scad_cv$lambda.min
coef_scad   <- coef(scad_cv$fit, lambda = lambda_scad)[-1]
names(coef_scad) <- paste0("X", 1:p)

cat("Lambda óptimo SCAD (CV 10-fold):", round(lambda_scad, 5), "\n")
cat("Coeficientes SCAD:\n")
print(round(coef_scad, 4))

cat("Lambda min (SCAD):", round(scad_cv$lambda.min, 5), "\n")
cat("Variables activas en lambda.min:", sum(round(coef_scad, 4) != 0), "\n")

# Tabla SCAD (mismo formato que tabla LASSO)
tabla_scad <- data.frame(
  Variable         = paste0("X", 1:p),
  "Beta Verdadero" = beta_verdadero,
  SCAD             = round(coef_scad, 4),
  Seleccionado     = ifelse(round(coef_scad, 4) != 0, "Sí", "No"),
  check.names = FALSE
)
print(tabla_scad)

tabla_scad_tex <- xtable(
  tabla_scad,
  caption = "Coeficientes estimados por SCAD en $\\hat{\\lambda}$ (validación cruzada 10-fold) comparados con los valores verdaderos.",
  label   = "tab:C3:scad",
  digits  = c(0, 0, 0, 4, 0)
)
sink(file.path(DIR_FIGS, "tabla_scad.tex"))
print(tabla_scad_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_scad.tex\n\n")

# ── Tabla comparativa conjunta: LASSO + Adaptive LASSO + SCAD ─────────────
# Colorear filas de variables relevantes en verde
sel_lasso  <- ifelse(round(coef_lasso,  4) != 0, "Sí", "No")
sel_alasso <- ifelse(round(coef_alasso, 4) != 0, "Sí", "No")
sel_scad   <- ifelse(round(coef_scad,   4) != 0, "Sí", "No")

vars_nombres <- paste0("X", 1:p)


vars_color <- ifelse(vars_nombres %in% vars_relevantes,
                     paste0("\\rowcolor{green!25}", vars_nombres),
                     vars_nombres)

tabla_reg_conjunta <- data.frame(
  Variable          = vars_color,
  "$\\beta$ verd."  = beta_verdadero,
  "LASSO"           = round(coef_lasso,  4),
  "Sel. LASSO"      = sel_lasso,
  "Ad. LASSO"       = round(coef_alasso, 4),
  "Sel. Ad. LASSO"  = sel_alasso,
  "SCAD"            = round(coef_scad,   4),
  "Sel. SCAD"       = sel_scad,
  check.names = FALSE
)

tabla_reg_tex <- xtable(
  tabla_reg_conjunta,
  caption = "Comparación de coeficientes estimados por LASSO, Adaptive LASSO y SCAD en $\\hat{\\lambda}_{\\min}$ (validación cruzada 10-fold). Las filas en verde corresponden a las variables verdaderamente relevantes del modelo.",
  label   = "tab:C3:regularizacion_comparacion",
  digits  = c(0, 0, 0, 4, 0, 4, 0, 4, 0)
)
sink(file.path(DIR_FIGS, "tabla_regularizacion_conjunta.tex"))
print(tabla_reg_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_regularizacion_conjunta.tex\n\n")

# =============================================================================
# FUNCIONES AUXILIARES PARA LAS FIGURAS DE REGULARIZACIÓN
# =============================================================================
# Paleta: un color distinto por variable, fijo y reproducible.
# Las variables relevantes tienen colores saturados y línea gruesa;
# las de ruido tienen colores apagados y línea fina.
# Usamos la misma paleta en las tres figuras y en las tablas LaTeX.
PALETA <- c(
  "#e41a1c",  # X1  — rojo        (relevante)
  "#aaaaaa",  # X2  — gris medio  (ruido)
  "#377eb8",  # X3  — azul        (relevante)
  "#bbbbbb",  # X4  — gris claro  (ruido)
  "#cccccc",  # X5  — gris claro  (ruido)
  "#dddddd",  # X6  — gris muy claro (ruido)
  "#4daf4a",  # X7  — verde       (relevante)
  "#bbbbbb",  # X8  — gris claro  (ruido)
  "#cccccc",  # X9  — gris claro  (ruido)
  "#dddddd"   # X10 — gris muy claro (ruido)
)
# Nombre de color LaTeX correspondiente (para la columna de color en tablas)
PALETA_LATEX <- c(
  "\\cellcolor[HTML]{E41A1C}\\textcolor{white}{rojo}",    # X1
  "\\cellcolor[HTML]{AAAAAA}{gris}",                      # X2
  "\\cellcolor[HTML]{377EB8}\\textcolor{white}{azul}",    # X3
  "\\cellcolor[HTML]{BBBBBB}{gris}",                      # X4
  "\\cellcolor[HTML]{CCCCCC}{gris}",                      # X5
  "\\cellcolor[HTML]{DDDDDD}{gris}",                      # X6
  "\\cellcolor[HTML]{4DAF4A}\\textcolor{white}{verde}",   # X7
  "\\cellcolor[HTML]{BBBBBB}{gris}",                      # X8
  "\\cellcolor[HTML]{CCCCCC}{gris}",                      # X9
  "\\cellcolor[HTML]{DDDDDD}{gris}"                       # X10
)
LWDS_VAR <- c(2, 0.7, 2, 0.7, 0.7, 0.7, 2, 0.7, 0.7, 0.7)
VARS_REL <- c("X1", "X3", "X7")

# Función: dibuja el path de un objeto glmnet con colores por variable
# y etiquetas al final de cada curva.
# Eje X: -log(lambda) — izquierda = lambda grande (modelo simple),
#                        derecha  = lambda pequeño (modelo rico)
dibujar_path_glmnet <- function(fit, lambda_opt) {
  beta_mat <- as.matrix(fit$beta)
  noms     <- rownames(beta_mat)
  xl       <- -log(fit$lambda)     # -log: crece hacia la derecha
  
  yrange <- range(beta_mat) * c(ifelse(min(beta_mat) < 0, 1.12, 0.95), 1.08)
  
  plot(NA,
       xlim     = range(xl),
       ylim     = yrange,
       xlab     = expression(-Log(lambda)),
       ylab     = "Coefficients",
       cex.lab  = 1.2,
       cex.axis = 1.1)
  
  for (j in seq_len(nrow(beta_mat))) {
    lines(xl, beta_mat[j, ],
          col = PALETA[j], lwd = LWDS_VAR[j])
  }
  
  abline(v = -log(lambda_opt), lty = 2, col = "black", lwd = 1.4)
  
  x_label   <- max(xl)
  # glmnet ordena lambdas de mayor a menor → columna con lambda más pequeño
  # es la última columna de beta_mat
  coefs_fin <- beta_mat[, ncol(beta_mat)]
  for (j in seq_len(nrow(beta_mat))) {
    if (abs(coefs_fin[j]) > 0.05) {
      text(x_label, coefs_fin[j],
           labels = noms[j],
           col    = PALETA[j],
           cex    = 0.72, adj = c(1.1, 0.5), font = 2)
    }
  }
  
  n_act  <- apply(beta_mat != 0, 2, sum)
  cambio <- c(TRUE, diff(n_act) != 0)
  axis(3, at = xl[cambio], labels = n_act[cambio],
       cex.axis = 0.72, tcl = -0.3)
  mtext(expression(hat(lambda) ~ "óptimo (CV)"),
        side = 3, line = 0.3, cex = 0.72, col = "black",
        at   = -log(lambda_opt))
}

# Función: dibuja el path de un objeto ncvreg con la misma paleta y eje
# ncvreg también ordena lambdas de mayor a menor; -log da el mismo convenio
dibujar_path_ncvreg <- function(fit, lambda_opt) {
  beta_mat <- fit$beta[-1, ]
  noms     <- rownames(beta_mat)
  xl       <- -log(fit$lambda)     # -log: crece hacia la derecha
  
  yrange <- range(beta_mat) * c(ifelse(min(beta_mat) < 0, 1.12, 0.95), 1.08)
  
  plot(NA,
       xlim     = range(xl),
       ylim     = yrange,
       xlab     = expression(-Log(lambda)),
       ylab     = expression(hat(beta)),
       cex.lab  = 1.2,
       cex.axis = 1.1)
  
  for (j in seq_len(nrow(beta_mat))) {
    lines(xl, beta_mat[j, ],
          col = PALETA[j], lwd = LWDS_VAR[j])
  }
  
  abline(v = -log(lambda_opt), lty = 2, col = "black", lwd = 1.4)
  abline(h = 0, col = "grey85", lwd = 0.5)
  
  # Etiquetas al extremo derecho (última columna = lambda más pequeño)
  x_label   <- max(xl)
  coefs_fin <- beta_mat[, ncol(beta_mat)]
  for (j in seq_len(nrow(beta_mat))) {
    if (abs(coefs_fin[j]) > 0.05) {
      text(x_label, coefs_fin[j],
           labels = noms[j],
           col    = PALETA[j],
           cex    = 0.72, adj = c(1.1, 0.5), font = 2)
    }
  }
  
  n_act  <- apply(beta_mat != 0, 2, sum)
  cambio <- c(TRUE, diff(n_act) != 0)
  axis(3, at = xl[cambio], labels = n_act[cambio],
       cex.axis = 0.72, tcl = -0.3)
  mtext(expression(hat(lambda) ~ "óptimo (CV)"),
        side = 3, line = 0.3, cex = 0.72, col = "black",
        at   = -log(lambda_opt))
}

# Función: curva CV de ncvreg dibujada a mano sobre -log(lambda)
dibujar_cv_ncvreg <- function(cv_fit) {
  lambdas <- cv_fit$fit$lambda
  xl      <- -log(lambdas)         # -log: crece hacia la derecha
  ecm     <- cv_fit$cve
  ecm_se  <- cv_fit$cvse
  
  plot(xl, ecm, type = "n",
       xlab     = expression(-Log(lambda)),
       ylab     = "Mean-Squared Error",
       ylim     = range(c(ecm - ecm_se, ecm + ecm_se)) * c(0.95, 1.05),
       cex.lab  = 1.2,
       cex.axis = 1.1)
  segments(xl, ecm - ecm_se, xl, ecm + ecm_se, col = "grey70", lwd = 1)
  points(xl, ecm, pch = 19, col = "#d73027", cex = 0.65)
  abline(v = -log(cv_fit$lambda.min), lty = 2, col = "black", lwd = 1.4)
  
  n_act  <- apply(cv_fit$fit$beta[-1, ] != 0, 2, sum)
  cambio <- c(TRUE, diff(n_act) != 0)
  axis(3, at = xl[cambio], labels = n_act[cambio],
       cex.axis = 0.72, tcl = -0.3)
}

# =============================================================================
# FIGURAS DE REGULARIZACIÓN
# =============================================================================

# ── Figura comparativa de funciones de penalización ───────────────────────
guardar_pdf("fig_penalizaciones.pdf", ancho = 8, alto = 5)
par(mar = c(4.5, 4.5, 3, 1.5), family = "serif")

beta_seq <- seq(-3, 3, length.out = 500)
lambda   <- 1
a_scad   <- 3.7   # parámetro SCAD estándar (Fan & Li 2001)

# Ridge: lambda * beta^2
pen_ridge <- lambda * beta_seq^2

# LASSO: lambda * |beta|
pen_lasso <- lambda * abs(beta_seq)

# SCAD: penalización por tramos (Fan & Li 2001)
pen_scad <- sapply(beta_seq, function(b) {
  ab <- abs(b)
  if (ab <= lambda) {
    lambda * ab
  } else if (ab <= a_scad * lambda) {
    -(ab^2 - 2 * a_scad * lambda * ab + lambda^2) / (2 * (a_scad - 1))
  } else {
    lambda^2 * (a_scad + 1) / 2
  }
})

plot(beta_seq, pen_lasso, type = "l", lwd = 2, col = "#2166ac",
     xlab = expression(beta), ylab = expression(rho(beta)),
     main = "Comparación de funciones de penalización",
     cex.main = 1.2, cex.lab = 1.0, ylim = c(0, max(pen_ridge) * 0.6))
lines(beta_seq, pen_ridge, lwd = 2, col = "#d73027", lty = 2)
lines(beta_seq, pen_scad,  lwd = 2, col = "#4daf4a", lty = 4)
legend("top", 
       legend = c(expression(paste("LASSO (", L[1], ")")),
                  expression(paste("Ridge (", L[2], ")")),
                  expression("SCAD")),
       col = c("#2166ac", "#d73027", "#4daf4a"),
       lty = c(1, 2, 4), lwd = 2, bty = "n", cex = 0.9)
dev.off()
cat(">> Guardada: fig_penalizaciones.pdf\n\n")


# ── Figura LASSO ─────────────────────────────────────────────────────────
guardar_pdf("fig_lasso_path.pdf", ancho = 11, alto = 6)
par(mfrow = c(1, 2), family = "serif")

par(mar = c(5, 5, 7, 1.5))
dibujar_path_glmnet(lasso_fit, lambda_opt)
title(main = "LASSO — trayectoria de coeficientes",
      line = 4.8, cex.main = 1.3, font.main = 2)

par(mar = c(5, 5.5, 5, 1.5))
plot(lasso_cv, cex.axis = 1.1, ylab = "", xlab = "")
title(main = "LASSO — validación cruzada (10-fold)",
      line = 3.5, cex.main = 1.3, font.main = 2)
mtext(expression(-Log(lambda)), side = 1, line = 3,   cex = 1.2)
mtext("Mean-Squared Error",     side = 2, line = 4,   cex = 1.2)
dev.off()
cat(">> Guardada: fig_lasso_path.pdf\n\n")

# ── Figura Adaptive LASSO ─────────────────────────────────────────────────
guardar_pdf("fig_alasso_path.pdf", ancho = 11, alto = 6)
par(mfrow = c(1, 2), family = "serif")

par(mar = c(5, 5, 7, 1.5))
dibujar_path_glmnet(alasso_fit, lambda_al)
title(main = "Adaptive LASSO — trayectoria de coeficientes",
      line = 4.8, cex.main = 1.3, font.main = 2)

par(mar = c(5, 5.5, 5, 1.5))
plot(alasso_cv, cex.axis = 1.1, ylab = "", xlab = "")
title(main = "Adaptive LASSO — validación cruzada (10-fold)",
      line = 3.5, cex.main = 1.3, font.main = 2)
mtext(expression(-Log(lambda)), side = 1, line = 3,   cex = 1.2)
mtext("Mean-Squared Error",     side = 2, line = 4,   cex = 1.2)
dev.off()
cat(">> Guardada: fig_alasso_path.pdf\n\n")

# ── Figura SCAD ───────────────────────────────────────────────────────────
guardar_pdf("fig_scad_path.pdf", ancho = 11, alto = 6)
par(mfrow = c(1, 2), family = "serif")

par(mar = c(5, 5, 7, 1.5))
dibujar_path_ncvreg(scad_cv$fit, lambda_scad)
title(main = "SCAD — trayectoria de coeficientes",
      line = 4.8, cex.main = 1.3, font.main = 2)

par(mar = c(5, 5.5, 5, 1.5))
dibujar_cv_ncvreg(scad_cv)
title(main = "SCAD — validación cruzada (10-fold)",
      line = 3.5, cex.main = 1.3, font.main = 2)
dev.off()
cat(">> Guardada: fig_scad_path.pdf\n\n")


# ── Tabla comparativa LASSO / Adaptive LASSO / SCAD con columna de color ─
tabla_reg <- data.frame(
  Color            = color_latex,
  Variable         = paste0("X", 1:p),
  Beta_verdadero   = beta_verdadero,
  LASSO            = round(coef_lasso,  4),
  "Adaptive LASSO" = round(coef_alasso, 4),
  SCAD             = round(coef_scad,   4),
  check.names = FALSE
)

cat("\nTabla comparativa de regularización:\n")
print(tabla_reg)

tabla_reg_tex <- xtable(
  tabla_reg,
  caption = "Comparación de los coeficientes estimados por LASSO, Adaptive LASSO y SCAD en sus respectivos $\\hat{\\lambda}$ óptimos (validación cruzada 10-fold) frente a los valores verdaderos. La columna de color identifica cada variable con el color de su curva en las figuras de trayectoria.",
  label   = "tab:C3:regularizacion",
  digits  = c(0, 0, 0, 0, 4, 4, 4)
)
sink(file.path(DIR_FIGS, "tabla_regularizacion.tex"))
print(tabla_reg_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity)
sink()
cat(">> Exportada: tabla_regularizacion.tex\n")

# Figura comparativa (barras agrupadas) — coeficientes por método
guardar_pdf("fig_comparacion_reg.pdf", ancho = 9, alto = 5)
metodos   <- c("Verdadero", "LASSO", "Ad. LASSO", "SCAD")
coefs_mat <- rbind(
  beta_verdadero,
  round(coef_lasso,  4),
  round(coef_alasso, 4),
  round(coef_scad,   4)
)
rownames(coefs_mat) <- metodos
colnames(coefs_mat) <- paste0("X", 1:p)

par(mar = c(5, 4.5, 4, 1), family = "serif")
barplot(coefs_mat,
        beside      = TRUE,
        col         = c("black", "#2166ac", "#4dac26", "#d7191c"),
        legend.text = metodos,
        args.legend = list(bty = "n", cex = 0.78, x = "topright",
                           inset = c(0, 0)),
        xlab    = "Covariable",
        ylab    = "Coeficiente estimado",
        main    = "Comparación: LASSO, Adaptive LASSO y SCAD",
        cex.main = 0.95,
        las     = 1,
        cex.names = 0.85,
        border  = "white")
abline(h = 0, lty = 1, col = "grey50")
dev.off()
cat(">> Guardada: fig_comparacion_reg.pdf\n\n")


# =============================================================================
# 3.4  BOOSTING LINEAL
# =============================================================================
cat("=== SECCIÓN 3.4: Boosting Lineal (L2-Boosting) ===\n")

boost_fit <- glmboost(Y ~ ., data = datos,
                      control = boost_control(mstop = 500, nu = 0.1))

# Selección del número óptimo de iteraciones (CV 10-fold, coherente con el resto)
set.seed(2024)
cv_boost  <- cvrisk(boost_fit,
                    folds = cv(model.weights(boost_fit),
                               type = "kfold", B = 10),
                    grid  = 1:500)
mstop_opt <- mstop(cv_boost)

boost_opt <- glmboost(Y ~ ., data = datos,
                      control = boost_control(mstop = mstop_opt, nu = 0.1))

cat("=== Resultados Boosting lineal ===\n")
cat("Iteraciones óptimas (CV 10-fold): m* =", mstop_opt, "\n")
cat("Tasa de aprendizaje: nu = 0.1\n")

coef_boost <- coef(boost_opt)[-1]
coef_boost_todos <- setNames(rep(0, p), paste0("X", 1:p))
coef_boost_todos[names(coef_boost)] <- coef_boost
cat("Coeficientes en m* =", mstop_opt, ":\n")
print(round(coef_boost_todos, 4))
cat("Variables activas en m*:", sum(round(coef_boost_todos, 4) != 0), "\n")
cat("Variables seleccionadas:", 
    paste(names(which(round(coef_boost_todos, 4) != 0)), collapse = ", "), "\n")

# Tabla Boosting transpuesta: filas = métrica, columnas = variables
vars_nombres <- paste0("X", 1:p)

# Nombres de columna con color para la tabla de boosting
col_names_color <- ifelse(
  vars_nombres %in% vars_relevantes,
  paste0("\\cellcolor{green!25}", vars_nombres),
  vars_nombres
)

fila_beta <- as.character(beta_verdadero)
fila_coef <- as.character(round(coef_boost_todos, 4))
fila_sel  <- ifelse(round(coef_boost_todos, 4) != 0, "Sí", "No")

# Aplicar color verde a las celdas de columnas relevantes en cada fila
aplicar_color_col <- function(valores, vars_nombres, vars_relevantes, color = "green!25") {
  ifelse(vars_nombres %in% vars_relevantes,
         paste0("\\cellcolor{", color, "}", valores),
         valores)
}

fila_beta_c <- aplicar_color_col(fila_beta, vars_nombres, vars_relevantes)
fila_coef_c <- aplicar_color_col(fila_coef, vars_nombres, vars_relevantes)
fila_sel_c  <- aplicar_color_col(fila_sel,  vars_nombres, vars_relevantes)

tabla_boost_t <- as.data.frame(
  rbind(fila_beta_c, fila_coef_c, fila_sel_c),
  stringsAsFactors = FALSE
)
colnames(tabla_boost_t) <- col_names_color
tabla_boost_t <- cbind(
  data.frame(Métrica = c("$\\beta$ verdadero", "Estimado", "Seleccionado"),
             check.names = FALSE),
  tabla_boost_t
)

tabla_boost_tex <- xtable(
  tabla_boost_t,
  caption = paste0("Coeficientes estimados por boosting lineal en $m^* = ",
                   mstop_opt,
                   "$ iteraciones. Las columnas en verde corresponden a las ",
                   "variables verdaderamente relevantes del modelo."),
  label = "tab:C3:boosting"
)
sink(file.path(DIR_FIGS, "tabla_boosting.tex"))
print(tabla_boost_tex, include.rownames = FALSE, booktabs = TRUE,
      sanitize.text.function = identity,
      sanitize.colnames.function = identity)
sink()
cat(">> Exportada: tabla_boosting.tex\n")

# Figura Boosting: trayectoria de coeficientes (izquierda) + curva CV (derecha)
pasos_tray <- c(1, 5, 10, 20, 50, 100, 200, 300, 500)

coef_tray <- matrix(0, nrow = length(pasos_tray), ncol = p)
rownames(coef_tray) <- pasos_tray
colnames(coef_tray) <- paste0("X", 1:p)

for (i in seq_along(pasos_tray)) {
  fit_tmp <- glmboost(Y ~ ., data = datos,
                      control = boost_control(mstop = pasos_tray[i], nu = 0.1))
  cc <- coef(fit_tmp)[-1]
  coef_tray[i, names(cc)] <- cc
}

guardar_pdf("fig_boosting_trayectoria.pdf", ancho = 11, alto = 6)
par(mfrow = c(1, 2), family = "serif")

# Gráfica izquierda: trayectoria de coeficientes
par(mar = c(5, 5, 5, 3))
matplot(pasos_tray, coef_tray,
        type  = "l", lty = 1,
        col   = PALETA,
        lwd   = c(2, 0.8, 2, 0.8, 0.8, 0.8, 2, 0.8, 0.8, 0.8),
        xlab  = "Número de iteraciones (m)",
        ylab  = "Coeficiente estimado",
        main  = "Boosting — evolución de coeficientes",
        cex.main = 1.3, cex.lab = 1.2, cex.axis = 1.1)
abline(v = mstop_opt, lty = 2, col = "black", lwd = 1.5)
abline(h = 0, col = "grey70", lwd = 0.8)

# Etiquetas al final de cada curva (última columna = más iteraciones)
x_label   <- max(pasos_tray)
coefs_fin <- coef_tray[nrow(coef_tray), ]
for (j in 1:p) {
  if (abs(coefs_fin[j]) > 0.15) {
    text(x_label, coefs_fin[j],
         labels = paste0("X", j),
         col    = PALETA[j],
         cex    = 0.75, adj = c(1.3, 0.5), font = 2)
  }
}

# Gráfica derecha: curva CV
par(mar = c(5, 5, 5, 1.5))
iters    <- 1:500
ecm_mean <- colMeans(cv_boost)
ecm_se   <- apply(cv_boost, 2, sd) / sqrt(nrow(cv_boost))

plot(iters, ecm_mean,
     type = "l", col = "black", lwd = 1.5,
     xlab = "Número de iteraciones (m)",
     ylab = "ECM (validación cruzada)",
     main = "Boosting — riesgo CV por iteración",
     cex.main = 1.3, cex.lab = 1.2, cex.axis = 1.1,
     ylim = range(c(ecm_mean - ecm_se, ecm_mean + ecm_se)) * c(0.97, 1.03))
polygon(c(iters, rev(iters)),
        c(ecm_mean + ecm_se, rev(ecm_mean - ecm_se)),
        col = adjustcolor("grey70", alpha.f = 0.4), border = NA)
lines(iters, ecm_mean, col = "black", lwd = 1.5)
abline(v = mstop_opt, lty = 2, col = "#d73027", lwd = 1.8)
legend("topright",
       legend = paste0("m* = ", mstop_opt),
       lty = 2, col = "#d73027", lwd = 1.8, bty = "n", cex = 0.85)
dev.off()


cat(">> Guardada: fig_boosting_trayectoria.pdf\n\n")


# =============================================================================
# FIGURA RESUMEN: Comparación de todos los métodos (heatmap de selección)
# =============================================================================
cat("=== FIGURA RESUMEN: todos los métodos ===\n")

# Construir coeficientes homogéneos para todos los métodos
coef_ols <- coef(modelo_completo)[-1]

coef_fwd <- rep(0, p); names(coef_fwd) <- paste0("X", 1:p)
cf       <- coef(modelo_step_fw)[-1]
coef_fwd[names(cf)] <- cf

coef_bth <- rep(0, p); names(coef_bth) <- paste0("X", 1:p)
cs       <- coef(modelo_step_bth)[-1]
coef_bth[names(cs)] <- cs

coef_boost_all <- rep(0, p); names(coef_boost_all) <- paste0("X", 1:p)
coef_boost_all[names(coef_boost)] <- coef_boost

resumen_mat <- rbind(
  Verdadero   = beta_verdadero,
  OLS         = round(coef_ols[paste0("X",1:p)], 3),
  Forward     = round(coef_fwd, 3),
  Stepwise    = round(coef_bth, 3),
  LASSO       = round(coef_lasso, 3),
  "Ad. LASSO" = round(coef_alasso, 3),
  SCAD        = round(coef_scad, 3),
  Boosting    = round(coef_boost_all, 3)
)
colnames(resumen_mat) <- paste0("X", 1:p)

guardar_pdf("fig_resumen_todos_metodos.pdf", ancho = 10, alto = 6)
par(mar = c(5, 7, 4, 2), family = "serif")

nm  <- nrow(resumen_mat)
image(1:p, 1:nm,
      t(abs(resumen_mat) > 0)[, nm:1],
      col  = c("white", "#2166ac"),
      axes = FALSE,
      xlab = "",
      ylab = "",
      main = "Variables seleccionadas por cada método (azul = seleccionada)")
axis(1, at = 1:p, labels = paste0("X", 1:p), las = 1, cex.axis = 0.85)
axis(2, at = 1:nm, labels = rev(rownames(resumen_mat)), las = 2, cex.axis = 0.85)
grid(nx = p, ny = nm, col = "grey85", lty = 1)
box()
dev.off()
cat(">> Guardada: fig_resumen_todos_metodos.pdf\n\n")


# =============================================================================
# RESUMEN FINAL EN CONSOLA
# =============================================================================
cat("\n========================================================\n")
cat("RESUMEN: variables seleccionadas por cada método\n")
cat("Verdadero: X1 (3), X3 (-2), X7 (1.5)\n")
cat("========================================================\n")

metodos_res <- list(
  OLS        = names(which(abs(coef(modelo_completo)[-1]) > 0.01)),
  Forward    = names(coef(modelo_step_fw))[-1],
  Backward   = names(coef(modelo_step_bw))[-1],
  Stepwise   = names(coef(modelo_step_bth))[-1],
  LASSO      = names(which(coef_lasso   != 0)),
  "Ad.LASSO" = names(which(coef_alasso  != 0)),
  SCAD       = names(which(coef_scad    != 0)),
  Boosting   = names(which(coef_boost   != 0))
)

for (nm in names(metodos_res)) {
  cat(sprintf("%-12s: %s\n", nm, paste(metodos_res[[nm]], collapse = ", ")))
}

cat("\n========================================================\n")
cat("Archivos generados en:", DIR_FIGS, "\n")
cat("  Tablas LaTeX:\n")
cat("    tabla_contrastes_t.tex\n")
cat("    tabla_contraste_F.tex\n")
cat("    tabla_best_subset.tex\n")
cat("    tabla_fwd_bwd_step.tex\n")
cat("    tabla_lasso.tex\n")
cat("    tabla_regularizacion.tex\n")
cat("    tabla_boosting.tex\n")
cat("  Figuras PDF:\n")
cat("    fig_pvalores_t.pdf\n")
cat("    fig_best_subset.pdf\n")
cat("    fig_best_subset_heatmap.pdf\n")
cat("    fig_forward_trayectoria.pdf\n")
cat("    fig_lasso_path.pdf\n")
cat("    fig_alasso_path.pdf\n")
cat("    fig_scad_path.pdf\n")
cat("    fig_comparacion_reg.pdf\n")
cat("    fig_boosting_trayectoria.pdf\n")
cat("    fig_resumen_todos_metodos.pdf\n")
cat("========================================================\n")