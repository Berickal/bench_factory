import coverage
import tempfile
import os

def merge_program(program: str, test_program: str) -> str:
    """
    Merges the main program and test program into a single string.
    Args:
        program (str): The main program code.
        test_program (str): The test code to be merged.
    Returns:
        str: The merged program code.
    """
    add_import = "import unittest\nimport pytest\n"
    full_program = add_import + extract_program(program) + "\n" + clean_test(test_program).strip()
    return full_program

def extract_program(program: str) -> str:
    """
    Extracts the main program code from a given string.
    """
    if "----" in program:
        program =  program.split("----", 1)[1]
    return program.split("----", 1)[0]

def clean_test(test_program: str) -> str:
    """
    Cleans the test program by ensuring it has a proper structure.
    Args:
        test_program (str): The test code to be cleaned.
    Returns:
        str: The cleaned test code.
    """
    if "class Test" in test_program:
        test = test_program.split("class Test", 1)[1]
        test = "class Test" + test
        # Fixed the typo: **nane** -> **name**
        if "if __name__" not in test:
            test += "\nif __name__ == '__main__':\n    unittest.main()\n"
        return test
    
    # Handle function-based tests
    if "def test" in test_program:
        test = test_program.split("def test", 1)[1]
        func_name = "test" + test.split("(", 1)[0].strip()
        test = "def test" + test
        if test.count(func_name) < 2:
            test += f"\n{func_name}()\n"
        return test
    
    return test_program

def get_code_coverage(program: str, test_program: str) -> dict:
    """
    Generates a code coverage report by merging the main program and test program.
    Args:
        program (str): The main program code.
        test_program (str): The test code to be merged.
    Returns:
        dict: Coverage information including percentage and detailed report.
    """
    # Merge the programs
    merged_code = merge_program(program, test_program)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(merged_code)
        temp_file_path = temp_file.name
    
    try:
        # Initialize coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Execute the merged code
        exec_globals = {}
        try:
            exec(compile(merged_code, temp_file_path, 'exec'), exec_globals)
        except Exception as e:
            print(f"Error executing code: {e}")
            return {"error": str(e), "coverage_percentage": 0}
        
        # Stop coverage and get results
        cov.stop()
        cov.save()
        
        # Generate coverage report
        coverage_data = cov.get_data()
        analysis = cov.analysis2(temp_file_path)
        
        if analysis:
            executed_lines = set(analysis[1])  # Lines that were executed
            missing_lines = set(analysis[2])   # Lines that were not executed
            total_lines = len(executed_lines) + len(missing_lines)
            
            if total_lines > 0:
                coverage_percentage = (len(executed_lines) / total_lines) * 100
            else:
                coverage_percentage = 0
            
            return {
                "coverage_percentage": round(coverage_percentage, 2),
                "executed_lines": len(executed_lines),
                "total_lines": total_lines,
                "missing_lines": list(missing_lines),
                "executed_lines_list": list(executed_lines),
                "temp_file_path": temp_file_path
            }
        else:
            return {"error": "No coverage data available", "coverage_percentage": 0}
            
    except Exception as e:
        return {"error": f"Coverage analysis failed: {str(e)}", "coverage_percentage": 0}
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def get_detailed_coverage_report(program: str, test_program: str) -> str:
    """
    Generates a detailed, human-readable coverage report.
    Args:
        program (str): The main program code.
        test_program (str): The test code to be merged.
    Returns:
        str: Detailed coverage report.
    """
    coverage_info = get_code_coverage(program, test_program)
    
    if "error" in coverage_info:
        return f"Coverage Analysis Failed: {coverage_info['error']}"
    
    report = f"""
Code Coverage Report
====================
Coverage Percentage: {coverage_info['coverage_percentage']}%
Executed Lines: {coverage_info['executed_lines']}
Total Executable Lines: {coverage_info['total_lines']}

"""
    
    if coverage_info['missing_lines']:
        report += f"Lines Not Covered: {sorted(coverage_info['missing_lines'])}\n"
    else:
        report += "All lines covered!\n"
    
    return report

def run_coverage_with_file(program_file: str, test_file: str) -> dict:
    """
    Run coverage analysis on separate program and test files.
    Args:
        program_file (str): Path to the main program file.
        test_file (str): Path to the test file.
    Returns:
        dict: Coverage information.
    """
    try:
        with open(program_file, 'r') as f:
            program_code = f.read()
        with open(test_file, 'r') as f:
            test_code = f.read()
        
        return get_code_coverage(program_code, test_code)
    
    except FileNotFoundError as e:
        return {"error": f"File not found: {str(e)}", "coverage_percentage": 0}
    except Exception as e:
        return {"error": f"Error reading files: {str(e)}", "coverage_percentage": 0}