# TODO list

* Implementation
  * batch_size = 100
  * L = 32
  * lr = 10e-3 (no decay)
  * random/det uniform projection on [0,pi]
  

* Writing
  * Sec.1: intro blabla
  * Sec.2: generative models
    * VAE (Benoit)
    * WAE (Benoit)
    * limits and improvments?
  * Sec.3: SW: derivation and implementation
    * theoretical bounds
  * Sec.4: Experiments
    * follow https://arxiv.org/pdf/1903.12436.pdf
    * MNIST (32,32,1), SVHN, CelebA, cifar10    
    * Projection ablation
      * compare loss + reconstruction score for different num of projections (non random?)
      * compare loss + reconstruction score for different projection distribution (fixed num of projections)
    * Auto-encoder ablation (not latent reg)
      * compare reconstruction score (MSE, FID) for (beta-)VAE/WAE L2/ WAE SW
    * Generative models
      * compare rec/samples with (beta-)VAE/WAE l2 using MSE and FID
