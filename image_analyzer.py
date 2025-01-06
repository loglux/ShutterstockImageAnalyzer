import time
from ollama import Client
from pydantic import BaseModel
import pandas as pd
import os
from pathlib import Path

# Define the schema for structured output using Pydantic
class ImageAnalysisResult(BaseModel):
    description: str
    keywords: list[str]
    categories: list[str]
    editorial: bool
    mature_content: bool
    illustration: bool


class ImageAnalyzer:
    ALLOWED_CATEGORIES = {
        "Abstract", "Animals/Wildlife", "Arts", "Backgrounds/Textures", "Beauty/Fashion",
        "Buildings/Landmarks", "Business/Finance", "Celebrities", "Education", "Food and drink",
        "Healthcare/Medical", "Holidays", "Industrial", "Interiors", "Miscellaneous", "Nature",
        "Objects", "Parks/Outdoor", "People", "Religion", "Science", "Signs/Symbols",
        "Sports/Recreation", "Technology", "Transportation", "Vintage"
    }

    def __init__(self, model="llama3.2-vision", base_url="http://localhost:11434/"):
        self.model = model
        self.base_url = base_url
        self.client = Client(host=base_url)

    def analyze_image(self, image_path, prompt=None, advanced_options=None, hint=None):
        try:
            # Define default prompt if none is provided
            if not prompt:
                prompt = (
                    """
                    Analyze this image and provide the following details:
                    1. Provide a concise image caption under 200 characters for Shutterstock. Avoid introductory phrase, be direct and descriptive. Avoid assumptions or guesses. If possible, specify the exact name of the object, landmark, or location (e.g., "Eiffel Tower" instead of "tower" or "landmark"; "Bald Eagle" instead of "bird"). Use specific terms for identifiable entities or features visible in the image, avoiding overly generic descriptions. Add emotional, engaging language to highlight the beauty, atmosphere, or unique character of the scene.
                    2. Generate no fewer than 7 and up to 50 unique and relevant keywords describing the image.  
                        - Focus on terms that are highly relevant to the image content and avoid overly generic words.  
                        - Use synonyms and related terms (e.g., "gull", "seagull", "waterbird") to diversify the keywords.  
                        - Avoid repeating the same concept unnecessarily unless it adds value.
                    3. Choose one or two categories that best match the image. 
                    - Do not generate more than two categories.  
                    - Strictly choose one or two categories from the provided list. Do not modify, combine, or create additional categories.
                    - If only one category applies, leave the second blank.  
                    - Example: "Nature" or "Buildings/Landmarks, Nature".
                    Available categories:
                    - Abstract
                    - Animals/Wildlife
                    - Arts
                    - Backgrounds/Textures
                    - Beauty/Fashion
                    - Buildings/Landmarks
                    - Business/Finance
                    - Celebrities
                    - Education
                    - Food and drink
                    - Healthcare/Medical
                    - Holidays
                    - Industrial
                    - Interiors
                    - Miscellaneous
                    - Nature
                    - Objects
                    - Parks/Outdoor
                    - People
                    - Religion
                    - Science
                    - Signs/Symbols
                    - Sports/Recreation
                    - Technology
                    - Transportation
                    - Vintage
                    4. Based on the visual content of the image, classify it as **commercial** or **editorial** based on the following criteria:
                    - **Commercial**:
                      - The image looks generic and polished, making it suitable for advertising or promotional use.
                      - It does NOT show visible logos, brand names, or trademarks.
                      - It does NOT feature clearly recognizable individuals, private properties, or artworks unless they are generic or unidentifiable.
                      - The scene appears intentionally staged or directed for professional purposes.
                    - **Editorial**:
                      - The image captures a real-life moment, event, or public place without significant staging.
                      - It may show visible logos, brand names, trademarks, recognizable individuals, or properties.
                      - The image feels spontaneous or candid, representing authentic, unscripted moments.
                      - It may illustrate cultural, social, or historical significance, or document a notable event or place.
                    5. Indicate if the image contains **Mature Content**:
                        - **Yes**: The image contains nudity, sexual themes, violence, or any content that could be considered inappropriate for a general audience.
                        - **No**: The image does not contain any of the above elements and is suitable for all audiences.
                    6. Indicate if the image qualifies as an **Illustration**:
                        - **Yes**: The image is created digitally, manually drawn, or heavily edited to include artistic or conceptual elements that are not photographic.
                        - **No**: The image is a straightforward photograph with no significant artistic manipulation.

                    Return the result in the following JSON format:
                    {
                        "description": "A brief descriptive text for the image.",
                        "keywords": ["keyword1", "keyword2", "..."],
                        "categories": ["category1", "category2"],
                        "editorial": true/false,
                        "mature_content": true/false,
                        "illustration": true/false
                    }
                    """
                )

            # if a hint is provided:
            if hint:
                prompt = f"{hint}\n\n{prompt}"
                print(f"Added hint to the prompt: {hint}")

            # Prepare the request payload
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_path]
                    }
                ],
                "format": ImageAnalysisResult.model_json_schema(),  # Pass the schema
                "options": {
                    # "repeat_last_n": 128, # randomly chosen, the default is 64
                    "num_ctx": 4096,
                    "num_predict": 600,  # low value causes JSON errors.
                    "top_k": 250,  # should increase the diversity of keywords
                    "repeat_penalty": 1.1,  # Starting with 1.2 and more reduces a number of keywords below 7
                    "temperature": 0.7,
                    "top_p": 0.9  # 0.9-1.0 should be OK, starting with 0.8 and low produces irrelevant keywords
                }
            }

            if advanced_options:
                data["options"].update(advanced_options.get("options", {}))

            print("Sending request to the model...")
            # print("Data sent:", json.dumps(data, indent=4))  # Debug: Show request payload
            response = self.client.chat(
                model=self.model,
                messages=data["messages"],
                format=data["format"],
                options=data["options"]
            )

            # Parse the JSON response
            return ImageAnalysisResult.model_validate_json(response.message.content)

        except Exception as e:
            return f"An error occurred: {e}"

    # @staticmethod
    def save_to_csv(self, results, image_path, file_path):
        """
        Save analysis results to a CSV file.

        Args:
            results (ImageAnalysisResult): Parsed analysis results.
            image_path (str): Name or path of the image file.
            file_path (str): Path to the CSV file.
        """
        # remove unsupported categories
        filtered_categories = self.filter_categories(results.categories)
        # Prepare data to append
        row = {
            # "Filename": image_path.strip(),
            "Filename": os.path.basename(image_path.strip()),
            "Description": results.description.strip(),
            "Keywords": ", ".join(results.keywords).strip(),
            # "Categories": ", ".join(results.categories).strip(),
            "Categories": ", ".join(filtered_categories).strip(),
            "Editorial": "yes" if results.editorial else "no",
            "Mature content": "yes" if results.mature_content else "no",
            "Illustration": "yes" if results.illustration else "no",
        }

        # Debug - print the size of description:
        # print(f"Description size: {len(results.description.strip())} characters")

        # Ensure correct column order
        column_order = [
            "Filename", "Description", "Keywords", "Categories", "Editorial",
            "Mature content", "Illustration"
        ]

        # Check if file exists
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Add missing columns with empty values
                for col in column_order:
                    if col not in df.columns:
                        df[col] = ""
                df = df[column_order]  # Reorder columns
            except Exception as e:
                print(f"Error reading the existing CSV file: {e}")
                df = pd.DataFrame(columns=column_order)
        else:
            # Create a new DataFrame with the required columns
            df = pd.DataFrame(columns=column_order)

        # Append the new row
        new_row_df = pd.DataFrame([row], columns=column_order)  # Ensure the new row has the correct column order
        df = pd.concat([df, new_row_df], ignore_index=True)

        # Save back to CSV
        df.to_csv(file_path, index=False, encoding="utf-8")

        print(f"Data saved to {file_path}")

    def start_analysis(self, image_path, file_path, prompt=None, advanced_options=None, hint=None):
        """
        Analyze an image and save the results to a CSV file.
        Args:
        :param image_path (str): Path to the image file.
        :param file_path (str): Path to the CSV file.
        :param promp (str, optional): Prompt to use for analysis. Defaults to None.
        :param advanced_options (dict, optional): Advanced options for the analysis. Defaults to None.
        """
        # Analyze the image
        result = self.analyze_image(image_path, prompt, advanced_options, hint=hint)

        # Ensure result is structured before proceeding
        if isinstance(result, ImageAnalysisResult):
            print("Analysis Result:", result)
            self.save_to_csv(result, image_path, file_path)
        else:
            print("Failed to analyze image:", result)

    def process_images_in_directory(self, directory_path, file_path, prompt=None, advanced_options=None, recursive=True, hint=None):
        """
        Search and process all images in a directory and subdirectories.
        Args:
            directory_path (str): Path to the directory containing images.
            file_path (str): Path to the CSV file to save results.
            prompt (str, optional): Prompt to use for analysis. Defaults to None.
            advanced_options (dict, optional): Advanced options for the analysis. Defaults to None.
            recursive (bool, optional): Search recursively in subdirectories. Defaults to True.
        """
        # Create object Path for the directory
        directory = Path(directory_path)

        if not directory.is_dir():
            print(f"Error: Directory not found: {directory_path}")
            return

        # Image searching mask
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        images = directory.rglob("*") if recursive else directory.glob("*")

        # Image file filter
        image_files = [file for file in images if file.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No images found in directory: {directory_path}")
            return

        print(f"Found {len(image_files)} images in directory: {directory_path}")

        # Processing images one by one
        for image_path in image_files:
            print(f"Processing: {image_path}")
            self.start_analysis(str(image_path), file_path, prompt, advanced_options, hint=hint)
            time.sleep(0.5)

    #@staticmethod
    def filter_categories(self, categories):
        """
        Filter out categories that are not in ALLOWED_CATEGORIES.
        Args:
            categories (list): Category list to filter.
        Returns:
            list: Filtered category list.
        """
        # return [category for category in categories if category in self.ALLOWED_CATEGORIES]
        return [category.strip().title() for category in categories if category.strip().title() in self.ALLOWED_CATEGORIES]

    @staticmethod
    def evaluate_prompt_compliance(csv_file_path, desc_max=200, key_min=7, key_max=50, category_count=2):
        if not os.path.exists(csv_file_path):
            return {"error": f"File not found: {csv_file_path}"}

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        if df.empty:
            return {"error": "CSV file is empty"}

        # Handling empty values in columns
        df["Categories"] = df["Categories"].fillna("")
        df["Description"] = df["Description"].fillna("")
        df["Keywords"] = df["Keywords"].fillna("")

        # Check the length of descriptions
        desc_lengths = df["Description"].str.strip().str.len()
        desc_compliance = (desc_lengths <= desc_max).mean() * 100

        # Check the number of keywords
        key_counts = df["Keywords"].apply(lambda x: len(str(x).split(", ")) if pd.notnull(x) else 0)
        key_min_compliance = (key_counts >= key_min).mean() * 100
        key_max_compliance = (key_counts <= key_max).mean() * 100

        # Check the number of categories
        category_compliance = (
                df["Categories"]
                .apply(lambda x: len(str(x).split(", ")) in [1, 2] if pd.notnull(x) else False)
                .mean() * 100
        )

        # Check the uniqueness of descriptions
        unique_descriptions = df["Description"].nunique()
        description_uniqueness = (unique_descriptions / len(df)) * 100

        # Check the repetition of opening phrases
        start_phrases = df["Description"].str.split().str[:5].str.join(" ")
        duplicate_starts = start_phrases.duplicated(keep=False).mean() * 100

        # Return the summary
        return {
            "description_compliance": desc_compliance,
            "keyword_min_compliance": key_min_compliance,
            "keyword_max_compliance": key_max_compliance,
            "category_compliance": category_compliance,
            "description_uniqueness": description_uniqueness,
            "duplicate_start_phrases": duplicate_starts,
        }

# Example usage
if __name__ == "__main__":
    analyzer = ImageAnalyzer()

    # Path to the image
    # image_path = r"D:\PycharmProjects\Lab\ShutterstockImageAnalyzer\DSC_1895.JPG"
    image_directory_path = r"path/to/images/directory"
    csv_file_path = "shutterstock.csv"

    # Add an optional hint before the prompt, to give a clue to the model
    hint = None
    # hint = "White Park Bay, Atlantic Ocean"

    #analyzer.start_analysis(image_path, prompt=None, advanced_options=None)
    analyzer.process_images_in_directory(image_directory_path, csv_file_path, prompt=None, advanced_options=None, recursive=False, hint=None)


