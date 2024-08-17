import streamlit as st
from index import semantic_search
from prompt_eng import generate_text, bold_unique_terms

# Set the title of the Streamlit app
st.title("Text Generator")

# Provide a brief description of the app's functionality
st.write("Text Generator uses the GenAI technology to generate the Business Scenario, Benefits and Description for a particular requirement of a user.")

# Create a text input field for the user to enter a requirement
txt=st.text_input("Enter a requirement")

# Call the `semantic_search` function to find the top 3 relevant items based on the user's input
top_3 = semantic_search(txt)

# Define the unique terms that need to be bolded in the generated text
unique_terms = {"Business Scenario:", "Benefits:", "Description:"}

# Generate the text based on the top 3 search results and the user's input
# Then bold the unique terms in the generated text
out = bold_unique_terms(generate_text(top_3,txt),unique_terms)

# Create a button labeled "SUBMIT" that, when clicked, will display the generated text
if st.button("SUBMIT")==True:
    # Display the generated text in the Streamlit app
    st.write(out)