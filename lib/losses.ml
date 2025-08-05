(* Loss functions and their derivatives for neural networks *)
open Lacaml.D

(* Mean Squared Error: (1/2) * mean((y_pred - y_true)^2) *)
let mean_squared_error (y_pred: Mat.t) (y_true: Mat.t) : float =
  let diff = Mat.sub y_pred y_true in
  let rows, cols = Mat.dim1 diff, Mat.dim2 diff in
  let total_elements = float (rows * cols) in
  let sum_squared = ref 0.0 in
  for i = 1 to rows do
    for j = 1 to cols do
      let d = diff.{i,j} in
      sum_squared := !sum_squared +. d *. d
    done
  done;
  0.5 *. (!sum_squared /. total_elements)

(* MSE gradient: (y_pred - y_true) / batch_size *)
let mse_grad (y_pred: Mat.t) (y_true: Mat.t) : Mat.t =
  let diff = Mat.sub y_pred y_true in
  let batch_size = float (Mat.dim1 diff) in
  Mat.map (fun x -> x /. batch_size) diff

(* Cross Entropy: -mean(sum(y_true * log(y_pred + epsilon))) *)
let cross_entropy (y_pred: Mat.t) (y_true: Mat.t) : float =
  let rows, cols = Mat.dim1 y_pred, Mat.dim2 y_pred in
  let batch_size = float rows in
  let epsilon = 1e-15 in
  let total_loss = ref 0.0 in
  for i = 1 to rows do
    for j = 1 to cols do
      let pred = y_pred.{i,j} +. epsilon in
      let true_val = y_true.{i,j} in
      total_loss := !total_loss +. true_val *. log pred
    done
  done;
  -.(!total_loss /. batch_size)

(* Cross Entropy gradient: (y_pred - y_true) / batch_size *)
let cross_entropy_grad (y_pred: Mat.t) (y_true: Mat.t) : Mat.t =
  let diff = Mat.sub y_pred y_true in
  let batch_size = float (Mat.dim1 diff) in
  Mat.map (fun x -> x /. batch_size) diff
