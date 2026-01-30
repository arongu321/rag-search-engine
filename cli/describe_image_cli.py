import argparse
import mimetypes
from google.genai import types
from lib.google_gemini_client import client, model
def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", type=str, help="Path to the image to describe")
    parser.add_argument("--query", type=str, help="Query to rewrite based on image")
    
    args = parser.parse_args()
    
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img = f.read()
    
    prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
    """
    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip()
    ]
    response = client.models.generate_content(
        model=model,
        contents=parts
    )
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()