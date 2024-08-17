import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """Preprocess the text by lowercasing, removing punctuation, and removing stopwords.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text with lowercase characters, no punctuation, and stopwords removed.
    """
    # Convert the entire text to lowercase to standardize it
    text = text.lower()

     # Remove punctuation from the text to focus on the actual words
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Get the set of English stopwords from the NLTK library
    stop_words = set(stopwords.words('english'))\

    # Tokenize the text into individual words
    word_tokens = word_tokenize(text)

    # Filter out stopwords from the tokenized text
    filtered_text = [word for word in word_tokens if word not in stop_words]

    # Join the remaining words back into a single string
    return ' '.join(filtered_text)

def remove_spl_char(text):
    """Remove special characters from text.

    Args:
        text (str): The input text containing special characters.

    Returns:
        str: The text with special characters removed.
    """
    # Encode text to ASCII, ignore non-ASCII characters, then decode back to UTF-8
    return text.encode('ascii', 'ignore').decode('utf-8')

def data_filter(data_dict):
    """Filter data by removing special characters.

    Args:
        data_dict (dict): A dictionary where keys and values are strings.

    Returns:
        dict: A new dictionary with special characters removed from all values.
    """
    # Apply `remove_spl_char` to each value in the dictionary to clean the text
    return {key: remove_spl_char(value) for key,value in data_dict.items()}