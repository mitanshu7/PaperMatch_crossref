# Import required libraries
import re
from datetime import datetime
from functools import cache
from urllib.parse import quote_plus

import gradio as gr
import numpy as np
from dotenv import dotenv_values
from mixedbread import Mixedbread
from pymilvus import MilvusClient
import backoff
import requests
################################################################################
# Configuration

# Get current year
current_year = str(datetime.now().year)

# Define Milvus client
milvus_client = MilvusClient("http://localhost:19530")

# Load Model
# Model to use for embedding
model_name = "mixedbread-ai/mxbai-embed-large-v1"

# Import secrets
config = dotenv_values(".env")

# Setup mxbai client
mxbai_api_key = config["MXBAI_API_KEY"]
mxbai = Mixedbread(api_key=mxbai_api_key)


################################################################################
# Function to extract DOI from a given text
def extract_doi(text:str):
    
    # https://www.crossref.org/blog/dois-and-matching-regular-expressions/
    pattern = re.compile(r"10.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

    match = pattern.search(text)
    
    if match:
        return match.group()
    else:
        return None


################################################################################
# Function to search crossref by DOI
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=requests.exceptions.RequestException,
    jitter=backoff.full_jitter,
    max_tries=3
)
def search_doi(doi: str):
    
    # URL encode the doi
    doi = quote_plus(doi)

    # Define the endpoint
    crossref_api_url = f"https://api.crossref.org/works/{doi}"

    response = requests.get(crossref_api_url)

    response_json = response.json()

    return response_json.get("message").get('abstract')



################################################################################
# Function to convert dense vector to binary vector
def dense_to_binary(dense_vector: np.ndarray) -> bytes:
    return np.packbits(np.where(dense_vector >= 0, 1, 0)).tobytes()


# Function to embed text
@cache
def embed(text: str) -> np.ndarray | bytes:

    # Call the MixedBread.ai API to generate the embedding
    result = mxbai.embed(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=text,
            normalized=True,
            encoding_format="ubinary",
            dimensions=1024,
            )

    # Convert the embedding to a numpy array of uint8 encoding and then to bytes
    embedding = np.array(result.data[0].embedding, dtype=np.uint8).tobytes()

    return embedding


################################################################################
# Single vector search


def search(vector: np.ndarray, limit: int, filter: str = "") -> list[dict]:
    # Logic for converting the filter to a valid format
    if filter == "This Year":
        filter = f"year == {int(current_year)}"
    elif filter == "Last 5 Years":
        filter = f"year >= {int(current_year) - 5}"
    elif filter == "Last 10 Years":
        filter = f"year >= {int(current_year) - 10}"
    elif filter == "All":
        filter = ""

    result = milvus_client.search(
        collection_name="crossref",  # Collection to search in
        data=[vector],  # Vector to search for
        limit=limit,  # Max. number of search results to return
        output_fields=[
            "DOI",
            "vector",
            "title",
            "abstract",
            "author",
            "month",
            "year",
            "URL",
        ],  # Output fields to return
        filter=filter,  # Filter to apply to the search
    )

    # returns a list of dictionaries with id and distance as keys
    return result[0]


################################################################################
# Function to fetch paper details of all results
def fetch_all_details(search_results: list[dict]) -> str:
    # Initialize an empty string to store the cards
    cards = ""

    for search_result in search_results:
        paper_details = search_result["entity"]

        # chr(10) is a new line character, replace to avoid formatting issues
        card = f"""
## [{paper_details["title"]}]({paper_details["URL"]})
> **{paper_details["author"]}** | _{paper_details["month"]} {paper_details["year"]}_ \n
{paper_details["abstract"]} \n
***
"""

        cards += card

    return cards


################################################################################


# Function to handle the UI logic
def predict(
    input_text: str, limit: int = 5, increment: int = 5, filter: str = ""
) -> tuple[str, gr.update, int]:
    # Check if input is empty
    if input_text == "":
        raise gr.Error("Please provide either a DOI or an abstract.", 10)

    # Define extra outputs to pass
    # This hack shows the load_more button once the search has been made
    show_load_more = gr.update(visible=True)
    show_date_filter = gr.update(visible=True)

    # This variable is used to increment the search limit when the load_more button is clicked
    new_limit = limit + increment

    # Extract doi, if any
    doi = extract_doi(input_text)

    # When doi is found in input text
    if doi:
        # Search if doi is already in database
        id_in_db = milvus_client.get(collection_name="crossref", ids=[doi])
        
        # If the doi is already in database
        if bool(id_in_db):
            # Get the bytes of a binary vector
            abstract_vector = id_in_db[0]["vector"][0]

        # If the doi is not already in database
        else:
            # Search crossref for paper details
            abstract = search_doi(doi)
            
            # If crossref returns an abstract
            if abstract:
                # Embed abstract
                abstract_vector = embed(abstract)
            else:
                raise gr.Error(f"Crossref did not return any abstract for doi: {doi}", 10)

    # When arxiv id is not found in input text, treat input text as abstract
    # Embed abstract
    abstract_vector = embed(input_text)

    # Search database
    search_results = search(abstract_vector, limit, filter)

    # Gather details about the found papers
    all_details = fetch_all_details(search_results)

    return all_details, show_load_more, show_date_filter, new_limit


################################################################################

# Variable to store contact information
contact_text = """
<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
    <h3>Crafted with ❤️ by <a href="https://mitanshu7.github.io" target="_blank">Mitanshu Sukhwani</a></h3>
    <h4>Discover more at <a href="https://bio.papermatch.me" target="_blank">PaperMatchBio</a></h4>
</div>
"""

# Examples to display
examples = ["2401.07215", "Smart TV and privacy"]

# Show total number of entries in database
num_entries = format(
    milvus_client.get_collection_stats(collection_name="crossref")["row_count"],
    ",",
)

# Create a back to top button
back_to_top_btn_html = """
<button id="toTopBtn" onclick="'parentIFrame' in window ? window.parentIFrame.scrollTo({top: 0, behavior:'smooth'}) : window.scrollTo({ top: 0 })">
    <a style="color:#6366f1; text-decoration:none;">&#8593;</a> <!-- Use the ^ character -->
</button>"""

# CSS for the back to top button
style = """
#toTopBtn {
    position: fixed;
    bottom: 10px;
    right: 10px; /* Adjust this value to position it at the bottom-right corner */
    height: 40px; /* Increase the height for a better look */
    width: 40px; /* Set a fixed width for the button */
    font-size: 20px; /* Set font size for the ^ icon */
    border-color: #e0e7ff; /* Change border color using hex */
    background-color: #e0e7ff; /* Change background color using hex */
    text-align: center; /* Align the text in the center */
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 50%; /* Make it circular */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); /* Add shadow for better visibility */
}

#toTopBtn:hover {
    background-color: #c7d4ff; /* Change background color on hover */
}
"""

# Markdown for about page
about_markdown = """
**PaperMatch** is a semantic search engine. Unlike regular search engines that match keywords in a query, semantic search engines convert text into vectors — essentially lists of numbers — using an *embedding model* (a type of neural network). These vectors aim to capture the **semantics** (meaning) of the text.

Because numbers can be compared (`a > b`), we can **indirectly compare text** by comparing their corresponding vectors. This is the core idea behind PaperMatch.

**PaperMatch** converts the abstract of an arXiv paper into a vector and performs a similarity search over a corpus of other papers.

---

## Guide to Using PaperMatch

### 🔍 Search by arXiv ID

* Enter the arXiv identifier (e.g., `1706.03762`) to search for similar papers.
* You can also paste the full arXiv URL — PaperMatch will automatically extract the ID.

### 📝 Search by Text

* Enter natural language describing the kind of paper you're looking for.
* Keep in mind: since the system matches by **semantics** (not keywords), exact terms may not appear in the results — it's the **meaning** that matters.
"""

################################################################################
# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(
        font=gr.themes.GoogleFont("Helvetica"),
        font_mono=gr.themes.GoogleFont("Roboto Mono"),
    ),
    title="PaperMatch",
    css=style,
    analytics_enabled=False,
) as demo:
    # Title and Subtitle
    gr.HTML(
        '<h1><a href="https://papermatch.me" style="font-weight: bold; text-decoration: none;">PaperMatch</a></h1>'
    )
    gr.HTML("<h3> Discover Relevant Research, Instantly ⚡</h3>")

    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            placeholder=f"Search {num_entries} papers on Crossref",
            autofocus=True,
            submit_btn=True,
            show_label=False,
        )

    with gr.Row():
        # Add the date filter
        with gr.Column(scale=4):
            date_filter = gr.Dropdown(
                label="Filter by Year",
                choices=["This Year", "Last 5 Years", "Last 10 Years", "All"],
                value="All",
                visible=False,
                multiselect=False,
                allow_custom_value=False,
                filterable=False,
            )

        # # Add sorting options
        # with gr.Column(scale=1):

    # Define the initial page limit
    page_limit = gr.State(5)

    # Define the increment for the "Load More" button
    increment = gr.State(5)

    # Define new page limit
    new_page_limit = gr.State(page_limit.value + increment.value)

    # Output section, displays the search results
    output = gr.Markdown(
        label="Related Papers",
        latex_delimiters=[{"left": "$", "right": "$", "display": False}],
        padding=True,
    )

    # Hidden by default, appears after the first search
    load_more_button = gr.Button("More results ⬇️", visible=False)

    # Event handler for the input text box, triggers the search function
    input_text.submit(
        predict,
        [input_text, page_limit, increment, date_filter],
        [output, load_more_button, date_filter, new_page_limit],
        api_name="search",
    )

    # Event handler for the date filter dropbox
    date_filter.change(
        predict,
        [input_text, page_limit, increment, date_filter],
        [output, load_more_button, date_filter, new_page_limit],
        api_name=False,
    )

    # Event handler for the "Load More" button
    load_more_button.click(
        predict,
        [input_text, new_page_limit, increment, date_filter],
        [output, load_more_button, date_filter, new_page_limit],
        api_name=False,
    )

    # Example inputs
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=[output, load_more_button, date_filter, new_page_limit],
        fn=predict,
        label="Try:",
        run_on_click=True,
        cache_examples=False,
    )

    # Back to top button
    gr.HTML(back_to_top_btn_html)

    # Attribution
    gr.HTML(contact_text)

with demo.route("About", "/about"):
    # Title and Subtitle
    gr.HTML(
        '<h1><a href="https://papermatch.me" style="font-weight: bold; text-decoration: none;">PaperMatch</a></h1>'
    )
    gr.HTML("<h3> Discover Relevant Research, Instantly ⚡</h3>")

    # The about text
    gr.Markdown(about_markdown, padding=True)

    # Attribution
    gr.HTML(contact_text)


################################################################################

if __name__ == "__main__":
    
    demo.launch(server_port=7870, favicon_path="logo.png", show_api=False, pwa=True)