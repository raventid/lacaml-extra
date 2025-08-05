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

(* Tanh activation function: tanh(x) element-wise *)
let tanh (x: Mat.t) : Mat.t =
  Mat.map (fun v -> Stdlib.tanh v) x

(* Tanh gradient: 1 - tanh(x)^2 *)
let tanh_grad (x: Mat.t) : Mat.t =
  let t = tanh x in
  Mat.map (fun t_val -> 1. -. t_val *. t_val) t

(* Softmax activation function: row-wise softmax for batches *)
let softmax (x: Mat.t) : Mat.t =
  let rows, cols = Mat.dim1 x, Mat.dim2 x in
  let result = Mat.create rows cols in
  Mat.fill result 0.0;
  for i = 1 to rows do
    (* Find max in row i for numerical stability *)
    let row_max = ref x.{i,1} in
    for j = 2 to cols do
      if x.{i,j} > !row_max then row_max := x.{i,j}
    done;
    
    (* Compute exp(x - max) and sum *)
    let exp_sum = ref 0.0 in
    for j = 1 to cols do
      let exp_val = exp (x.{i,j} -. !row_max) in
      result.{i,j} <- exp_val;
      exp_sum := !exp_sum +. exp_val
    done;
    
    (* Normalize by sum *)
    for j = 1 to cols do
      result.{i,j} <- result.{i,j} /. !exp_sum
    done
  done;
  result

(* Softmax gradient: y_pred - y_true (for use with cross-entropy) *)
let softmax_grad (y_pred: Mat.t) (y_true: Mat.t) : Mat.t =
  Mat.sub y_pred y_true
