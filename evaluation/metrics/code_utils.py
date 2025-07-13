def remove_duplicate(res : str, ref_ouput : str):
    """
    Clean the response by capturing the relevant lines from the reference output.

    Args:
        res (str): The response output from the LLM.
        ref_ouput (str): The reference output to compare against.
    Returns:
        str: The cleaned response containing only the relevant lines.
    """
    
    ref_lines = ref_ouput.splitlines()
    ref_out_lines = res.splitlines()

    ## In case of multiple answers, we take the first one
    try:
        if ref_out_lines.count(ref_out_lines[0]) > 1:
            res_ = ref_out_lines[0] + "\n"
            for idx in range(len(ref_out_lines) + 1):
                if ref_out_lines[idx] != ref_out_lines[0] and idx != 0:
                    res_ += ref_out_lines[idx] + "\n"
                elif idx != 0:
                    break
            return res_.replace("User: ", "").replace("Assistant: ", "").strip()
    except IndexError:
        pass

    ## In case of Blablating, we identify the code based on the first line of the reference output
    try:
        f_index = ref_out_lines.index(ref_lines[0])
    except ValueError:
        f_index = -1
        for idx in range(len(ref_out_lines)):
            if "import " in ref_out_lines[idx] or ("def" in ref_out_lines[idx] and ":" in ref_out_lines[idx]):
                f_index = idx
    if f_index != -1:
        res = "\n".join(ref_out_lines[f_index:f_index+len(ref_lines)+1])
        if len(res.splitlines()[-1].split()) > 5 :
            return "\n".join(ref_out_lines[f_index:f_index+len(ref_lines)]).replace("User: ", "").replace("Assistant: ", "").strip()
    return res.replace("User: ", "").replace("Assistant: ", "").strip()


def clean_code(response: str) -> str:
    """
    Clean the code response by removing leading and trailing whitespace.
    
    Args:
        response (str): The code response to clean.
        
    Returns:
        str: The cleaned code response.
    """
    if "```python" in response:
        response = response.split("```python")[1]
    if "```" in response:
        response = response.split("```")[0]
    return response.strip()