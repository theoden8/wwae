# TODO list

* SW implementation
  * Check benoit/antoine versions match
  * Check if either version is actually correct
    * Possible test: images with one pixel with value 1, and 0 elsewere
  * Convert into tf code and merge with code base
* Implement NN from RAE
* First experiments
  * celebA
  * quick exp. with small num of hyperparameters
  * compare with VAE and WAE l2sq using MSE, FID score
* Experiments
  * follow https://arxiv.org/pdf/1903.12436.pdf
  * Auto-encoder ablation (not latent reg)
    * compare reconstruction score (MSE, FID) for (beta-)VAE/WAE L2/ WAE SW
  * Projection ablation
    * compare loss + reconstruction score for different num of projections (non random?)
    * compare loss + reconstruction score for different projection distribution (fixed num of projections)
  * Main experiments
    * MNIST (32,32,1), SVHN, CelebA, cifar10
    * compare with (beta-)VAE/WAE l2 using MSE and FID
