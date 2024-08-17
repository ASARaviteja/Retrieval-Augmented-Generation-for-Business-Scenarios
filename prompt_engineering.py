import openai
from config import openai_api_key

# Initialize the OpenAI client with the provided API key
client = openai.OpenAI(api_key=openai_api_key)

# Function to generate text based on provided content and user input
def generate_text(content, user_input):
    """
    Generate a comprehensive business scenario, benefits, and description based on provided content and user input.

    Args:
    content: A list of dictionaries containing requirements, business scenarios, benefits, and descriptions.
    user_input: A string representing the specific user requirement.

    Returns:
    A string containing the generated text divided into three sections: Business Scenario, Benefits, and Description.
    """
    if content:
        # Define the prompt text for the OpenAI API
        Prompt_text = f"""
		1. Please provide a comprehensive business scenario, outlining the context, challenges, and objectives related to this business requirement "[Requirement]".
        2. Detail the benefits that implementing this solution would bring to the organization.
        3. Describe technical details on how this solution would be implemented, including key functionalities, data integration requirements, user interface considerations, and any relevant technologies or methodologies.
        4. The output should be in paragraph only.
        5. There should be three sections: Business Scenario, Benefits, and Description.
        6. The Business Scenario should be in the perspective of the user starting with "As a user".
        7. Summarize the Business Scenario in 50 words.
        8. Summarize the Benefits in 50 words.
        9. Summarize the Description in 100-150 words.
        10. Ensure clarity and coherence in each section to provide a comprehensive understanding.

		EXAMPLE 1:
		 "
        Requirement:		{content[0].get('Requirement', None)}
		Business Scenario: 	{content[0].get('Business Scenario', None)}
		Benefit:	        {content[0].get('Benefit', None)}
		Description 		{content[0].get('Description', None)}
		"
		Example 2 :
        "
        Requirement:		{content[1].get('Requirement', None)}
		Business Scenario:	{content[1].get('Business Scenario', None)}
		Benefit:        	{content[1].get('Benefit', None)}
		Description   		{content[1].get('Description', None)}
		"
		Example 3 :
        "
		Business Scenario:	{content[2].get('Business Scenario', None)}
		Benefit:        	{content[2].get('Benefit', None)}
		Requirement:		{content[2].get('Requirement', None)}
		Description:   		{content[2].get('Description', None)}
		"
		"""

		# Create the modified content based on user input
        modified_content = f"""Generate functional specifications for the below given requirements.
                            REQUIREMENT: "{user_input}"""

		# Send a request to the OpenAI API to generate a response
        response2 = client.chat.completions.create(
            model="gpt-3.5-turbo",
			messages=[{"role":"system","content":Prompt_text},
					  {"role":"user","content":modified_content}],
			temperature=0.4, # Control the randomness of the output
            max_tokens=700,  # Limit the maximum number of tokens in the response
            top_p=0.1,       # Limit the diversity of the output
			frequency_penalty=0,
			presence_penalty=0,
			stop=None
            )

		# Return the generated text from the response
        return response2.choices[0].message.content

# Function to bold specific terms in the text
def bold_unique_terms(text, terms):
    """
    Bold specific terms in the given text.

    Args:
    text: The original text in which terms need to be bolded.
    terms: A set of terms that need to be bolded in the text.

    Returns:
    The modified text with the specified terms bolded.
    """
    for term in terms:
        text = text.replace(term, f"**{term}**")
    return text