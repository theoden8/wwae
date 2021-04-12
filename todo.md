- Fix compat bug in tf1 v. tf2 to get the same results
  - [] remove all tf.compat.v1 in the core of the scripts and put compat.v1 at the begining
  - [] try with/without disable_v2_behavior()

- [] WEMD: find correct weights on coeffs and check if a reweighting makes any sense wrt W1/OT/stability

- mnist experiments
  - [] 3 diagonal positions of digit '1'
  - [] compare results depending on the spacing between positions in the diagonal: 
  if the 3 positions are close/afar from each other, does it transcribe on the embedding?
  - [] add perturbation to the positions in the diagonal
  - [] subsample mnist digits: possibility to have more positions or add padding to remove periodic border effect from fft
  - [] 3 diagonal positions of digit '3'
  - [] 4 or + diagonal positions for a single digit
  - [] 3 diagonal positions for more than one digit
  - [] more complex positions than diagonal
