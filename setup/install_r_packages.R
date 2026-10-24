# Install the R packages that conda-forge does not build for osx-arm64.
# Run inside the fedgm env, after `mamba env create -f environment.yml`:
#   conda activate fedgm && Rscript setup/install_r_packages.R
# Packages go into the env's R library; already-installed ones are skipped.

pkgs <- c("rgenoud", "Matching")  # rgenoud first: GenMatch's optimizer
repo <- "https://cloud.r-project.org"

missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, repos = repo, lib = .libPaths()[1])
}

# install.packages only warns on failure; make a failed install an error.
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop(sprintf("R package '%s' failed to install", p))
  }
  message(sprintf("%s %s", p, as.character(packageVersion(p))))
}
