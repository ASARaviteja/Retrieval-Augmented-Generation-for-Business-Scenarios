import docx

def get_data_from_table(table_name):
     """Extract data from a Word table.

    Args:
        table_name (docx.table.Table): A `Table` object from the `docx` library.

    Returns:
        list of list of str: A 2D list where each sublist represents a row in the table and each element is a cell's text.
    """
     """Extract data from a Word table."""
    # Extract text from each cell in the table and organize it into a 2D list
     return [[cell.text for cell in row.cells] for row in table_name.rows]

def convert_list_to_dict(lst):
    """Convert a list of lists to a dictionary.

    Args:
        lst (list of list): A list of lists where each sublist contains two elements; the first is the key and the second is the value.

    Returns:
        dict: A dictionary where the first element of each sublist is a key and the second element is the corresponding value.
    """
    # Convert the list of lists to a dictionary, assuming each sublist has exactly two elements: key and value
    return {item[0]:item[1] for item in lst}

def get_data_from_doc(doc):
    """Get data from the first table in a Word document.

    Args:
        doc (str): The file path to the Word document.

    Returns:
        dict: A dictionary where the keys and values are extracted from the first table in the document.
    """
    # Open the Word document
    document = docx.Document(doc)

    # Extract data from the first table in the document
    d = get_data_from_table(document.tables[0])

    # Convert the list of lists from the table into a dictionary
    return convert_list_to_dict(d)