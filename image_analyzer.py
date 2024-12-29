import json
from ollama import Client
from pydantic import BaseModel
import pandas as pd
import os

# Define the schema for structured output using Pydantic
class ImageAnalysisResult(BaseModel):
    description: str
    keywords: list[str]
    categories: list[str]
    editorial: bool
    mature_content: bool
    illustration: bool

class ImageAnalyzer:
    def __init__(self, model="llama3.2-vision", base_url="http://localhost:11434/"):
        self.model = model
        self.base_url = base_url
        self.client = Client(host=base_url)

    def analyze_image(self, image_path, prompt=None, advanced_options=None):
        try:
            # Define default prompt if none is provided
            if not prompt:
                prompt = (
                    """
                    Analyze this image and provide the following details:
                    1. Provide a descriptive text for the image, suitable for Shutterstock, with a maximum length of 200 characters. If possible, specify the exact name of the object (e.g., eagle, crane) rather than using broad term like **bird**.
                    2. Include **at least 7** and up to 50 unique and diverse keywords that are highly relevant to the image content, even if they are not directly mentioned in the description. Ensure to include synonyms (e.g., "gull", "seagull", "waterbird"), while avoiding contradictory or conflicting terms.
                    3. Two categories that best describe the image. Categories **must** be chosen strictly from the following list:

                    Abstract, Animals/Wildlife, Arts, Backgrounds/Textures, Beauty/Fashion, Buildings/Landmarks, 
                    Business/Finance, Celebrities, Education, Food and drink, Healthcare/Medical, Holidays, 
                    Industrial, Interiors, Miscellaneous, Nature, Objects, Parks/Outdoor, People, Religion, 
                    Science, Signs/Symbols, Sports/Recreation, Technology, Transportation, Vintage.

                    4. Based on the image content, classify it as **commercial** or **editorial**:
                        - **Commercial**: The image must meet the following requirements:
                            - All recognisable individuals must have a signed and valid model release.
                            - Recognisable private properties, artworks, or objects require property releases.
                            - No visible logos, trademarks, or brand names are present.
                            - The image is free of intellectual property restrictions and is suitable for promotional use.

                        - **Editorial**: The image meets one or more of the following conditions:
                            - No releases (model or property) are available for recognisable individuals or private properties.
                            - The image contains logos, trademarks, or brand names.
                            - The content documents a specific event, place, or public activity, or tells a story that is newsworthy or educational.
                            - The image has not been posed or directed by the photographer and represents an authentic moment in time.

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
                    "num_ctx": 4096, # didn't spot the difference
                    "num_predict": 200, # didn't spot the difference
                    "top_k": 150, # should increase the diversity of keywords
                    "repeat_penalty": 1.1, # starting with 1.2 and more reduce a number of keywords
                    "temperature": 0.5,
                    "top_p": 0.9 # 0.9-1.0 should be OK, starting with 0.8 and low produces irrelevant keywords
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

    @staticmethod
    def save_to_csv(results, image_path, file_path):
        """
        Save analysis results to a CSV file.

        Args:
            results (ImageAnalysisResult): Parsed analysis results.
            image_path (str): Name or path of the image file.
            file_path (str): Path to the CSV file.
        """
        # Prepare data to append
        row = {
            "Filename": image_path.strip(),
            "Description": results.description.strip(),
            "Keywords": ", ".join(results.keywords).strip(),
            "Categories": ", ".join(results.categories).strip(),
            "Editorial": "yes" if results.editorial else "no",
            "Mature content": "yes" if results.mature_content else "no",
            "Illustration": "yes" if results.illustration else "no",
        }

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
        df.to_csv(file_path, index=False)

        print(f"Data saved to {file_path}")

# Example usage
if __name__ == "__main__":
    analyzer = ImageAnalyzer()

    # Path to the image
    image_path = "DSC_8205.JPG"
    file_path = "shutterstock.csv"

    # Analyze the image
    result = analyzer.analyze_image(image_path)

    # Ensure result is structured before proceeding
    if isinstance(result, ImageAnalysisResult):
        print("Analysis Result:", result)
        analyzer.save_to_csv(result, image_path, file_path)
    else:
        print("Failed to analyze image:", result)
