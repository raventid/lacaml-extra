open Lacaml.D
open Lacaml_extra
open Alcotest
module Q = QCheck

(* Simple utility to check if two matrices are approximately equal *)
let mat_approx_equal ?(tol=1e-10) m1 m2 =
  let rows1, cols1 = Mat.dim1 m1, Mat.dim2 m1 in
  let rows2, cols2 = Mat.dim1 m2, Mat.dim2 m2 in
  if rows1 <> rows2 || cols1 <> cols2 then false
  else
    try
      for i = 1 to rows1 do
        for j = 1 to cols1 do
          let diff = abs_float (m1.{i,j} -. m2.{i,j}) in
          if diff > tol then raise Exit
        done
      done;
      true
    with Exit -> false

let mat_testable = testable (fun fmt m ->
  let rows, cols = Mat.dim1 m, Mat.dim2 m in
  Format.fprintf fmt "Mat(%dx%d)" rows cols) mat_approx_equal

(* Test for ReLU activation function *)
let test_relu_basic () =
  let input = Mat.of_array [| [| -1.0; 0.0; 2.0 |] |] in
  let expected = Mat.of_array [| [| 0.0; 0.0; 2.0 |] |] in
  let result = Activations.relu input in
  check mat_testable "relu basic test" expected result

let test_relu_batch () =
  let input = Mat.of_array [| 
    [| -2.0; 1.0; 0.0 |];
    [| 3.0; -1.5; 0.5 |]
  |] in
  let expected = Mat.of_array [| 
    [| 0.0; 1.0; 0.0 |];
    [| 3.0; 0.0; 0.5 |]
  |] in
  let result = Activations.relu input in
  check mat_testable "relu batch test" expected result

(* Fixed property test for ReLU - generates rectangular matrices *)
let relu_property =
  Q.Test.make ~count:10 ~name:"relu always non-negative"
    (Q.pair (Q.int_range 1 5) (Q.int_range 1 5))
    (fun (rows, cols) ->
      let matrix_array = Array.make_matrix rows cols 0.0 in
      let rng = Random.State.make_self_init () in
      for i = 0 to rows - 1 do
        for j = 0 to cols - 1 do
          matrix_array.(i).(j) <- Random.State.float rng 20.0 -. 10.0
        done
      done;
      let input = Mat.of_array matrix_array in
      let result = Activations.relu input in
      let result_rows, result_cols = Mat.dim1 result, Mat.dim2 result in
      let all_non_negative = ref true in
      for i = 1 to result_rows do
        for j = 1 to result_cols do
          if result.{i,j} < 0.0 then all_non_negative := false
        done
      done;
      !all_non_negative)

(* Test for ReLU gradient *)
let test_relu_grad_basic () =
  let input = Mat.of_array [| [| -1.0; 0.0; 2.0 |] |] in
  let expected = Mat.of_array [| [| 0.0; 0.0; 1.0 |] |] in
  let result = Activations.relu_grad input in
  check mat_testable "relu_grad basic test" expected result

(* Test for Sigmoid activation function *)
let test_sigmoid_basic () =
  let input = Mat.of_array [| [| 0.0; 2.0; -2.0 |] |] in
  let expected = Mat.of_array [| [| 0.5; 0.8807970779778823; 0.11920292202211757 |] |] in
  let result = Activations.sigmoid input in
  check mat_testable "sigmoid basic test" expected result

let test_sigmoid_batch () =
  let input = Mat.of_array [| 
    [| 0.0; 1.0 |];
    [| -1.0; 0.5 |]
  |] in
  let result = Activations.sigmoid input in
  let rows, cols = Mat.dim1 result, Mat.dim2 result in
  (* Check all values are between 0 and 1 *)
  let all_in_range = ref true in
  for i = 1 to rows do
    for j = 1 to cols do
      let v = result.{i,j} in
      if v < 0.0 || v > 1.0 then all_in_range := false
    done
  done;
  check bool "sigmoid values in range [0,1]" true !all_in_range

(* Test for Sigmoid gradient *)
let test_sigmoid_grad_basic () =
  let input = Mat.of_array [| [| 0.0; 2.0; -2.0 |] |] in
  let result = Activations.sigmoid_grad input in
  let rows, cols = Mat.dim1 result, Mat.dim2 result in
  (* Check all gradient values are positive (sigmoid gradient is always > 0) *)
  let all_positive = ref true in
  for i = 1 to rows do
    for j = 1 to cols do
      if result.{i,j} <= 0.0 then all_positive := false
    done
  done;
  check bool "sigmoid_grad values are positive" true !all_positive

(* Test for Tanh activation function *)
let test_tanh_basic () =
  let input = Mat.of_array [| [| 0.0; 1.0; -1.0 |] |] in
  let expected = Mat.of_array [| [| 0.0; 0.7615941559557649; -0.7615941559557649 |] |] in
  let result = Activations.tanh input in
  check mat_testable "tanh basic test" expected result

let test_tanh_batch () =
  let input = Mat.of_array [| 
    [| 0.0; 2.0 |];
    [| -2.0; 0.5 |]
  |] in
  let result = Activations.tanh input in
  let rows, cols = Mat.dim1 result, Mat.dim2 result in
  (* Check all values are between -1 and 1 *)
  let all_in_range = ref true in
  for i = 1 to rows do
    for j = 1 to cols do
      let v = result.{i,j} in
      if v < -1.0 || v > 1.0 then all_in_range := false
    done
  done;
  check bool "tanh values in range [-1,1]" true !all_in_range

(* Test for Tanh gradient *)
let test_tanh_grad_basic () =
  let input = Mat.of_array [| [| 0.0; 1.0; -1.0 |] |] in
  let result = Activations.tanh_grad input in
  let rows, cols = Mat.dim1 result, Mat.dim2 result in
  (* Check all gradient values are positive and <= 1 (tanh gradient is always in (0,1]) *)
  let all_valid = ref true in
  for i = 1 to rows do
    for j = 1 to cols do
      let v = result.{i,j} in
      if v <= 0.0 || v > 1.0 then all_valid := false
    done
  done;
  check bool "tanh_grad values in (0,1]" true !all_valid

(* Test for Softmax activation function *)
let test_softmax_basic () =
  let input = Mat.of_array [| [| 1.0; 2.0; 3.0 |] |] in
  let result = Activations.softmax input in
  let _, cols = Mat.dim1 result, Mat.dim2 result in
  (* Check that each row sums to 1.0 *)
  let sum = ref 0.0 in
  for j = 1 to cols do
    sum := !sum +. result.{1,j}
  done;
  let diff = abs_float (!sum -. 1.0) in
  check bool "softmax row sums to 1" true (diff < 1e-10)

let test_softmax_batch () =
  let input = Mat.of_array [| 
    [| 1.0; 2.0; 3.0 |];
    [| 0.0; 1.0; 0.0 |]
  |] in
  let result = Activations.softmax input in
  let rows, cols = Mat.dim1 result, Mat.dim2 result in
  (* Check that each row sums to 1.0 *)
  let all_sum_to_one = ref true in
  for i = 1 to rows do
    let sum = ref 0.0 in
    for j = 1 to cols do
      sum := !sum +. result.{i,j}
    done;
    let diff = abs_float (!sum -. 1.0) in
    if diff >= 1e-10 then all_sum_to_one := false
  done;
  check bool "softmax batch rows sum to 1" true !all_sum_to_one

let test_softmax_stability () =
  (* Test numerical stability with large values *)
  let input = Mat.of_array [| [| 1000.0; 1001.0; 1002.0 |] |] in
  let result = Activations.softmax input in
  (* Should not contain NaN or infinity *)
  let is_finite = ref true in
  let rows, cols = Mat.dim1 result, Mat.dim2 result in
  for i = 1 to rows do
    for j = 1 to cols do
      let v = result.{i,j} in
      if not (Float.is_finite v) then is_finite := false
    done
  done;
  check bool "softmax is numerically stable" true !is_finite

(* Test for Softmax gradient (commonly used with cross-entropy) *)
let test_softmax_grad_basic () =
  let y_pred = Mat.of_array [| [| 0.2; 0.3; 0.5 |] |] in
  let y_true = Mat.of_array [| [| 0.0; 0.0; 1.0 |] |] in
  let result = Activations.softmax_grad y_pred y_true in
  let expected = Mat.of_array [| [| 0.2; 0.3; -0.5 |] |] in
  check mat_testable "softmax_grad basic test" expected result

let relu_grad_property =
  Q.Test.make ~count:10 ~name:"relu_grad is 0 or 1"
    (Q.pair (Q.int_range 1 5) (Q.int_range 1 5))
    (fun (rows, cols) ->
      let matrix_array = Array.make_matrix rows cols 0.0 in
      let rng = Random.State.make_self_init () in
      for i = 0 to rows - 1 do
        for j = 0 to cols - 1 do
          matrix_array.(i).(j) <- Random.State.float rng 20.0 -. 10.0
        done
      done;
      let input = Mat.of_array matrix_array in
      let result = Activations.relu_grad input in
      let result_rows, result_cols = Mat.dim1 result, Mat.dim2 result in
      let all_binary = ref true in
      for i = 1 to result_rows do
        for j = 1 to result_cols do
          let v = result.{i,j} in
          if v <> 0.0 && v <> 1.0 then all_binary := false
        done
      done;
      !all_binary)

let activation_tests = [
  test_case "relu basic" `Quick test_relu_basic;
  test_case "relu batch" `Quick test_relu_batch;
  test_case "relu_grad basic" `Quick test_relu_grad_basic;
  test_case "sigmoid basic" `Quick test_sigmoid_basic;
  test_case "sigmoid batch" `Quick test_sigmoid_batch;
  test_case "sigmoid_grad basic" `Quick test_sigmoid_grad_basic;
  test_case "tanh basic" `Quick test_tanh_basic;
  test_case "tanh batch" `Quick test_tanh_batch;
  test_case "tanh_grad basic" `Quick test_tanh_grad_basic;
  test_case "softmax basic" `Quick test_softmax_basic;
  test_case "softmax batch" `Quick test_softmax_batch;
  test_case "softmax stability" `Quick test_softmax_stability;
  test_case "softmax_grad basic" `Quick test_softmax_grad_basic;
]

let property_tests = [
  QCheck_alcotest.to_alcotest relu_property;
  QCheck_alcotest.to_alcotest relu_grad_property;
]

let () =
  run "Lacaml Extra Tests" [
    ("Activations", activation_tests);
    ("Properties", property_tests);
  ]
