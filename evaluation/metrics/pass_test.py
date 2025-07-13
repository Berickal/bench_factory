from dataclasses import dataclass
import sys
import os
sys.path.append("../")
from evaluation.metrics.metric import Metric, PassTest
from models.task_gen import LLMGen
import tempfile
import subprocess
import time
from dataclasses import dataclass

from unittest.mock import patch
from io import StringIO
import concurrent.futures
from evaluation.metrics.code_utils import clean_code, remove_duplicate

@dataclass
class TestReport:
    passed : bool
    functional_error : bool
    runtime_error : bool
    message : str

def update_test_name(response : str, test : str) -> str:
    """
    Update the test name in the response code.
    
    Args:
        response (str): The code response from the LLM.
        test (str): The test to be updated.
        
    Returns:
        str: The updated code with the new test name.
    """
    function_name = response.rsplit('def ', 1)[-1].split('(', 1)[0].strip()
    test_name = test.split("assert", 1)[1].split("(", 1)[0].strip()
    if function_name not in test:
        test = test.replace(test_name, function_name)
    return test

def evaluate_quixbugs_instance(response : str, 
                               tests : str, 
                               timeout : int = 10, 
                               programming_language : str = "python", 
                               ref_output : str = "", 
                               hard_clean : bool = False) -> TestReport:
    response = clean_code(response)
    if hard_clean:
        response = remove_duplicate(response, ref_output)

    if programming_language == "python":
        tests = update_test_name(response, tests)
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
            combined_code = f"{response}\n\n# Tests\n{tests}"
            temp_file.write(combined_code)
            temp_file_path = temp_file.name
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return TestReport(
                    passed=True,
                    functional_error=False,
                    runtime_error=False,
                    message=f"All tests passed successfully.\nOutput: {result.stdout}"
                )
            else:
                if "AssertionError" in result.stderr or "assert" in result.err:
                    return TestReport(
                        passed=False,
                        functional_error=True,
                        runtime_error=False,
                        message=f"Test assertion failed: {result.stderr}"
                    )
                else:
                    return TestReport(
                        passed=False,
                        functional_error=False,
                        runtime_error=True,
                        message=f"Code execution error: {result.stderr}"
                    )
                    
        except subprocess.TimeoutExpired:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message="Test timed out"
            )
        except Exception as e:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message=f"Runtime error: {str(e)}"
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    else:
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message=f"Unsupported programming language: {programming_language}"
        )
    

def evaluate_mbbp_instance(response : str, tests : list[str], timeout : int = 10, ref_output : str = "") -> TestReport:
    """
    Evaluate a MBPP instance by running the provided code against the tests.
    
    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds.
        
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    return evaluate_quixbugs_instance(response, "\n".join(tests).replace('"None"', 'None'), timeout, programming_language="python", ref_output=ref_output)

def evaluate_human_eval_instance(response : str, tests : str, timeout : int = 10, ref_output : str = "") -> TestReport:
    """
    Evaluate a HumanEval instance by running the provided code against the tests.
    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds.
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    assert_tests = tests.split("def check(candidate):")[1]
    return evaluate_quixbugs_instance(response, assert_tests.replace("    ", ""), timeout, programming_language="python", ref_output=ref_output)




def run_code(code, test_input_lines):
    """
    Function to run the code with redirected input and output.
    """
    inputs_iter = iter(test_input_lines)
    fake_input = lambda: next(inputs_iter)
    fake_stdout = StringIO()

    with patch('builtins.input', fake_input), patch('sys.stdout', fake_stdout):
        exec(code, {})

    return fake_stdout.getvalue().strip()

def evaluate_condefect_instance(response : str, test_in : list[str], test_out : list[str], timeout : int = 10, ref_output : str = "", hard_clean : bool = False) -> TestReport:
    """
    Evaluate a ConDefect instance by running the provided code against the tests.
    
    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds.
        
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    response = clean_code(response)
    if hard_clean:
        response = remove_duplicate(response, ref_output)

    for t_in, t_out in zip(test_in, test_out):
        try:
            if not (os.path.exists(t_in) and os.path.exists(t_out)):
                raise FileNotFoundError(f"Test files {t_in} or {t_out} not found.")
            
            with open(t_in, 'r') as f:
                test_input_lines = f.readlines()
            
            with open(t_out, 'r') as f:
                test_output_lines = f.readlines()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_code, response, test_input_lines)
                try:
                    actual_output = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    return TestReport(
                        passed=False,
                        functional_error=False,
                        runtime_error=True,
                        message="Test timed out"
                    )
            expected_output = ''.join(test_output_lines).strip()
            if actual_output == expected_output:
                return TestReport(
                    passed=True,
                    functional_error=False,
                    runtime_error=False,
                    message="Test passed successfully."
                )
            else:
                return TestReport(
                    passed=False,
                    functional_error=True,
                    runtime_error=False,
                    message=f"Test failed: Expected '{expected_output}', got '{actual_output}'"
                )
        except Exception as e:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message=f"Runtime error: {str(e)}"
            )
    