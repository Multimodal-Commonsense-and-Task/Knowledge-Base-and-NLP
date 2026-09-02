import re

def _extract_milestone_completion(message: str):
    """
    Extracts milestone numbers from a string.
    Pattern: [MILE: <num>] (Case-insensitive, flexible spacing)
    """
    pattern = r"\[[Mm][Ii][Ll][Ee]:\s*(\d+)\s*\]"
    
    # Find all matches and convert the captured groups to integers
    matches = re.findall(pattern, message)
    
    return [int(num) for num in matches]

def _remove_milestone_tokens(message: str) -> str:
        pattern = r"\[[Mm][Ii][Ll][Ee]:\s*(\d+)\s*\]"
        cleaned_message = re.sub(pattern, "", message)
        return cleaned_message.strip()

if __name__ == "__main__":
    test_message = "We have completed the first step. [MILE: 1] Now moving to the next. [MILE: 2] blabla [MiLe:3] blabla [mile:  s  4   ]"
    milestones = _extract_milestone_completion(test_message)
    cleaned_message = _remove_milestone_tokens(test_message)
    
    print("Extracted Milestones:", milestones)
    print("Cleaned Message:", cleaned_message)