(lang dune 3.0)

(name lacaml-extra)

(package
 (name lacaml-extra)
 (version 0.1.0)
 (synopsis "High-level neural network utilities for OCaml, extending Lacaml")
 (description "Provides activations, gradients, and losses built on Lacaml's linear algebra.")
 (depends 
  dune
  lacaml
  alcotest
  qcheck
  qcheck-alcotest))
