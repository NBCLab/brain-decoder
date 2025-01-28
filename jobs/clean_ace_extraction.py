import os.path as op
import shutil
from glob import glob


def find_nth_occurrence(text, search_term, n):
    """
    Find the index of the nth occurrence of a search term in text.
    If nth occurrence doesn't exist but at least one does, returns the last found occurrence.
    Returns -1 if no occurrences found.
    """
    occurrences = []
    start = 0

    # Find all occurrences
    while True:
        index = text.lower().find(search_term.lower(), start)
        if index == -1:
            break
        occurrences.append(index)
        start = index + 1

    # If no occurrences found, return -1
    if not occurrences:
        return -1

    # If fewer occurrences than requested, return the last available one
    if len(occurrences) < n:
        return occurrences[-1]

    # Return the requested occurrence
    return occurrences[n - 1]


def extract_text(text, intro_occurrence=1):
    """
    Extracts text between nth occurrence of 'Introduction' and last occurrence of 'References'.
    Returns None if either section is not found.

    Parameters:
    text (str): The input text to process
    intro_occurrence (int): Which occurrence of Introduction to use (default: 2 for second occurrence)

    Returns:
    str: The extracted text between specified Introduction and last References, or None if not found
    """
    try:
        # Find the nth occurrence of "Introduction"
        start_idx = find_nth_occurrence(text, "introduction", intro_occurrence)
        if start_idx == -1:
            print("Introduction not found")
            # Look for Abstract
            start_idx = find_nth_occurrence(text, "abstract", intro_occurrence)
            if start_idx == -1:
                print("Abstract not found")
                start_idx = find_nth_occurrence(text, "background", intro_occurrence)
                if start_idx == -1:
                    print("Background not found")
                    start_idx = find_nth_occurrence(text, "summary", intro_occurrence)
                    if start_idx == -1:
                        print("Summary not found")
                        return None

        # Find Acknowledgments
        end_idx = text.lower().rfind("acknowledgments")
        if end_idx == -1:
            print("Acknowledgments not found")
            end_idx = text.lower().rfind("acknowledgements")
            if end_idx == -1:
                print("Acknowledgment not found")
                end_idx = text.lower().rfind("references")
                if end_idx == -1:
                    print("References not found")
                    end_idx = text.lower().rfind("reference")
                    if end_idx == -1:
                        print("Reference not found")
                        return None

        # Make sure References comes after Introduction
        if end_idx <= start_idx:
            print("References found before Introduction")
            start_idx = 0

        # Extract and return the text between these points
        return text[start_idx:end_idx].strip()

    except Exception as e:
        print(f"Error processing text: {e}")
        return None


data_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder/data/pubmed"
sorted_dirs = sorted(glob(op.join(data_dir, "*")))

dataset_dict = {}
for dset_dir in sorted_dirs:
    proc_dir = op.join(dset_dir, "processed")
    if not op.exists(proc_dir):
        continue

    pmid = op.basename(dset_dir)

    extract_dirs = sorted(glob(op.join(proc_dir, "*")))

    if len(extract_dirs) == 0:
        continue

    extracts = [op.basename(ext) for ext in extract_dirs]

    if ("ace" not in extracts) or ("pubget" in extracts):
        continue

    sel_dirs = op.join(proc_dir, "ace")

    extract = op.basename(sel_dirs)
    text_fn = op.join(sel_dirs, "text.txt")
    old_text_fn = op.join(sel_dirs, "text_orig.txt")

    if not op.exists(text_fn):
        continue

    print(f"Processing {dset_dir}")
    # Make copy of the original text copyin the file
    if not op.exists(old_text_fn):
        shutil.copy(text_fn, old_text_fn)

    with open(old_text_fn, "r") as file:
        body = file.read()

    if len(body) == 0:
        continue

    extracted_text = extract_text(body)

    # Write the extracted text back to the file
    with open(text_fn, "w") as file:
        file.write(extracted_text)
