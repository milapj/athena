import os
import sys
import json
import asyncio
from dotenv import load_dotenv
import PyPDF2  # or import pdfplumber

load_dotenv()

# Fixed imports - changed from relative to absolute
from dataset_builder import DatasetBuilder
from logic_tree.tree import LogicTree
from model import OpenAIModel
from validators.types.structure_validator import StructureValidator

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text_by_chars(text: str, chunk_size: int = 2000) -> list:
    """
    Splits the text into chunks, each with at most `chunk_size` characters.
    Returns a list of chunk strings.
    """
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

# class MockModel(Model):
#     """
#     A mock LLM for demonstration. 
#     Replace with your real model (e.g. OpenAIModel or HF model).
#     """
#     def inference(self, prompt, temperature=0.7):
#         # Return a dummy structure resembling an LLM response
#         return {
#             "choices": [
#                 {
#                     "message": {
#                         "content": f"Mocked response for prompt:\n{prompt[:100]}..."
#                     }
#                 }
#             ]
#         }



async def main():
    # 1) Read + chunk the file
    filename = "dataset/OTTOMAN TUBES PRIVATE LIMITED VS KARAN AUTOMOTIVES.pdf"
    
    # Extract text from PDF
    entire_text = extract_text_from_pdf(filename)
    
    chunks = chunk_text_by_chars(entire_text, chunk_size=2000)
    print(f"Split file into {len(chunks)} chunks.")

    # 2) Initialize a DatasetBuilder
    builder = DatasetBuilder()
    
    llm = OpenAIModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        engine="gpt-4o",   # or "text-davinci-003"
        api_endpoint="chat",      # or "completion"
        temperature=0.7,
        max_tokens=16000
    )

    # 4) Prepare a prompt function for each chunk
    #    We'll reuse the same prompt structure for all chunks.
    completion_prompt_fn = builder.create_completion_prompt(
        example_trees=[],
        example_node_completions=[],
        example_descriptions=[],
        intro="We are creating a logic tree for each chunk of the text file...",
        because_clause_after=-1,
        because_clause='Because, ',
        use_complex_facts=False
    )

    all_completed_trees = []
    complete_tree_str = []
    # 5) Iterate over each chunk
    for i, chunk_text in enumerate(chunks, start=1):
        print(f"[Chunk {i}/{len(chunks)}] Building logic tree...")

        # if(i == 20):
        #     break
        # Build a fresh logic tree skeleton
        logic_tree = builder.build_structure(
            depth=3,
            bf_factor={1: 2.0, 2: 1.0},
            chance_to_prune_all=0.3,
            chance_to_prune=0.3,
            root_nodes=[]
        )

        # Complete the logic tree with the chunk as 'description'
        completed_tree = builder.complete_structure(
            _tree=logic_tree,
            model=llm,
            description=chunk_text,
            completion_prompt_fn=completion_prompt_fn,
            max_retries_on_error=1,
            inplace=False,
            retry_model=llm,
            progress_bar=False,
            test_prompt=False,
            use_iterative_complete_v2=True,
            validators=[StructureValidator()],
        )
        
        complete_tree_str.append(completed_tree.print_for_gpt())

        # Convert to JSON-serializable
        completed_tree_json = completed_tree.to_json()
        all_completed_trees.append({
            "chunk_index": i,
            "chunk_text": chunk_text,
            "logic_tree": completed_tree_json
        })

    # 6) Store the combined results in a single JSON
    output_data = {
        "filename": filename,
        "total_chunks": len(chunks),
        "trees": all_completed_trees
    }
    
    with open("tree_string-2.txt", "w", encoding="utf-8" ) as f:
        for line in complete_tree_str:
            f.write(line)
            

    with open("logic_trees_for_chunks.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved all chunk-based trees to logic_trees_for_chunks.json")

if __name__ == "__main__":
    # If using Python 3.11+ or 3.10+ with the -m approach:
    # cd /home/biokami/github/grayscope
    # python -m src.grayscope.athena.main
    asyncio.run(main())
