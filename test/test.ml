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

(* Simple property test for ReLU without Mat.fold *)
let relu_property =
  Q.Test.make ~count:10 ~name:"relu always non-negative"
    (Q.list (Q.list (Q.float_range (-10.0) 10.0)))
    (fun matrix_list ->
      if List.length matrix_list = 0 || List.exists (fun row -> List.length row = 0) matrix_list then
        true  (* Skip empty matrices *)
      else
        let matrix_array = Array.of_list (List.map Array.of_list matrix_list) in
        let input = Mat.of_array matrix_array in
        let result = Activations.relu input in
        let rows, cols = Mat.dim1 result, Mat.dim2 result in
        let all_non_negative = ref true in
        for i = 1 to rows do
          for j = 1 to cols do
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
  let input = Mat.of_array [| [| 0.0; 1.0; -1.0 |] |] in
  let result = Activations.sigmoid_grad input in
  (* sigmoid_grad(x) = sigmoid(x) * (1 - sigmoid(x)) *)
  (* For x=0: sigmoid(0) = 0.5, so grad = 0.5 * 0.5 = 0.25 *)
  let expected_approx_zero = 0.25 in
  check (float 1e-10) "sigmoid_grad at zero" expected_approx_zero result.{1,1}

let relu_grad_property =
  Q.Test.make ~count:10 ~name:"relu_grad is 0 or 1"
    (Q.list (Q.list (Q.float_range (-10.0) 10.0)))
    (fun matrix_list ->
      if List.length matrix_list = 0 || List.exists (fun row -> List.length row = 0) matrix_list then
        true  (* Skip empty matrices *)
      else
        let matrix_array = Array.of_list (List.map Array.of_list matrix_list) in
        let input = Mat.of_array matrix_array in
        let result = Activations.relu_grad input in
        let rows, cols = Mat.dim1 result, Mat.dim2 result in
        let all_binary = ref true in
        for i = 1 to rows do
          for j = 1 to cols do
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
