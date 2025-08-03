(* Activation functions and their derivatives for neural networks *)
open Lacaml.D

(* ReLU activation function: max(0, x) element-wise *)
let relu (x: Mat.t) : Mat.t =
  Mat.map (fun v -> max 0. v) x

(* ReLU gradient: 1 if x > 0, 0 otherwise *)
let relu_grad (x: Mat.t) : Mat.t =
  Mat.map (fun v -> if v > 0. then 1. else 0.) x

(* Sigmoid activation function: 1 / (1 + exp(-x)) element-wise *)
let sigmoid (x: Mat.t) : Mat.t =
  Mat.map (fun v -> 1. /. (1. +. exp (-.v))) x

(* Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) *)
let sigmoid_grad (x: Mat.t) : Mat.t =
  let s = sigmoid x in
  Mat.map (fun s_val -> s_val *. (1. -. s_val)) s
