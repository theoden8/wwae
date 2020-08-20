# TODO list

* SW implementation
  * Check benoit/antoine versions match
  * Check if either version is actually correct
  * Convert into tf code and merge with code base
* Implement ResNet
  * check state-of-the-art or benchmark networks
* First experiments
  * celebA
  * quick exp. with small num of hyperparameters
  * compare with VAE and WAE using MSE, FID score
* Planification for final exp.
  * Auto-encoder ablation (not latent reg)
    * compare reconstruction score (MSE, any other?) for (beta-)VAE/WAE L2/ WAE SW
  * Projection ablation
    * compare loss + reconstruction score for different num of projections (non random?)
    * compare loss + reconstruction score for different projection distribution (fixed num of projections)
  * Main experiments
    * CelebA, cifar10, LSUN
    * compare with (beta-)VAE/WAE l2 using MSE and FID
