import config
import faiss
import json
import numpy as np
import pandas as pd
import pickle
from Data_Extraction import get_data_from_doc
from pre_processing import preprocess_text,data_filter
from sklearn.feature_extraction.text import TfidfVectorizer

def combine_dicts(dict1,dict2):
    """Combine two dictionaries.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary containing the combined key-value pairs from both dictionaries.
    """
    # Merge dict1 and dict2 into a new dictionary
    return {**dict1,**dict2}

def get_embeddings(text,vectorizer):
    """Get TF-IDF embeddings for the provided text.

    Args:
        text (list of str): A list of text strings to be vectorized.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer to transform the text.

    Returns:
        list: A list of TF-IDF feature vectors for the input text.
    """
    # Transform the text into TF-IDF vectors and convert them to dense lists
    vectors = vectorizer.transform(text)
    return vectors.todense().tolist()[0]

def create_embeddings(data_dict):
    """Create and save embeddings from the data dictionary.

    Args:
        data_dict (dict): A dictionary containing text data for creating embeddings.

    Returns:
        np.ndarray: The embeddings array created from the provided data.
    """
    # Convert the dictionary to a DataFrame with a single row
    df = pd.DataFrame(data_dict, index=[0])

    # Load the pre-trained TF-IDF vectorizer from a file
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    # Combine different text fields into a single text string for embedding
    combined_text = df["Requirement"]+df["Business Scenario"]+ df["Benefit"]+ df["Description"]

    # Generate embeddings for the combined text
    df["embeddings"] = [get_embeddings(combined_text.astype(str).tolist(),vectorizer)]

    # Save the DataFrame with embeddings to a pickle file
    df.to_pickle('data.pkl')

    # Stack all embeddings into a single numpy array
    embeddings = np.vstack(df['embeddings'])

    # Save the embeddings as a numpy binary file
    np.save("embeddings.npy", embeddings)

    return embeddings

def process_user_stories_requirements(start,end):
    """Process user stories and requirements from specified range.

    Args:
        start (int): The starting index for document processing.
        end (int): The ending index for document processing.
    """
    tests = []
    embed_dim = 791
    index = faiss.IndexFlatL2(embed_dim)
    for i in range(start,end+1):
        try:
            # Extract and filter text data from user story and requirement documents
            user_story = data_filter(get_data_from_doc(f"User stories/{i}_user_story.docx"))
            requirement = data_filter(get_data_from_doc(f"requirements/{i}_requirement.docx"))

            # Combine the two dictionaries into one
            dic = combine_dicts(requirement,user_story)

            # Add the combined dictionary to the list
            tests.append(dic)

            # Create embeddings for the combined text and add them to the FAISS index
            embeddings = create_embeddings(dic)
            index.add(embeddings)
        except Exception as e:
            # Print an error message if there is an issue processing a document
            print(f"Error processing document {i}: {e}")
            continue

    # Prepare the text data for saving
    text_data = {'texts': tests}

    # Save the FAISS index to the file specified in the config
    faiss.write_index(index, config.index_file)

    # Save the text data to a JSON file specified in the config
    with open(config.text_data, 'w') as file:
        json.dump(text_data, file)

def corpus(start,end):
    """Generate a corpus from the specified range of documents.

    Args:
        start (int): The starting index for document processing.
        end (int): The ending index for document processing.

    Returns:
        list: A list of preprocessed text data from the documents.
    """
    combined_texts=""
    lst = []
    for i in range(start,end+1):
        try:
            # Extract and filter text data from user story and requirement documents
            user_story = data_filter(get_data_from_doc(f"User stories/{i}_user_story.docx"))
            requirement = data_filter(get_data_from_doc(f"requirements/{i}_requirement.docx"))

            # Combine the two dictionaries into one
            combine_data = combine_dicts(requirement,user_story)

            # Concatenate the text fields into one string
            combined_texts += combine_data["Requirement"] + combine_data["Benefit"] + combine_data["Business Scenario"] + combine_data["Description"]

            # Preprocess the combined text and add it to the list
            lst.append(preprocess_text(combined_texts))
        except Exception as e:
            # Print an error message if there is an issue processing a document
            print(f"Error processing document {i}: {e}")
            continue

    return lst

def semantic_search(query):
    """Perform a semantic search on the indexed data.

    Args:
        query (str): The search query to be vectorized and compared against the indexed data.

    Returns:
        list: A list of documents (texts) that are similar to the query based on the semantic search.
    """
    # Load the pre-trained TF-IDF vectorizer from a file
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Get the embedding for the query text
    query_embed = get_embeddings([query],vectorizer)
    query_embed = np.vstack(query_embed).reshape(1,-1)

    # Load the FAISS index from the file specified in the config
    index = faiss.read_index(config.index_file)
    number = 3

    # Perform the search on the FAISS index
    distance,indices = index.search(query_embed, number)
    threshold = 0.5

    # Filter the results based on the distance threshold
    filtered_i = [ i for (d,i) in zip(distance[0],indices[0]) if d >= threshold]

    # Load the text data from the JSON file specified in the config
    with open(config.text_data, 'r') as file:
        text_data = json.load(file)

    # Return the filtered texts based on the indices
    return [text_data["texts"][i] for i in filtered_i]

# Generate the text corpus from documents in the specified range
text = corpus(1,20)

# Initialize and fit the TF-IDF vectorizer on the corpus text data
vectorizer = TfidfVectorizer()
vectorizer.fit(text)

# Save the trained TF-IDF vectorizer to a file
with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

# Process user stories and requirements in the specified range and create embeddings
process_user_stories_requirements(1,20)