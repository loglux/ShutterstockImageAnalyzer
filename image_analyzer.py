import json
from ollama import Client
import pandas as pd
import os
import re


class ImageAnalyzer:
    def __init__(self, model="llama3.2-vision", base_url="http://localhost:11434/"):
        self.model = model
        self.base_url = base_url
        self.client = Client(host=base_url)

    def analyze_image(self, image_path, prompt=None, advanced_options=None):
        try:
            if prompt:
                prompt_to_use = prompt
            else:
                prompt_to_use = (
                    """
                    Analyze this image and provide the following details:
                    1. A descriptive text for the image suitable for Shutterstock, up to 200 characters.
                    2. At least 7 unique keywords relevant to the image content. Up to 50 keywords.
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

                    Return the result in the following Markdown format:

                    # Description
                    A brief descriptive text about the image.

                    # Keywords
                    - keyword1
                    - keyword2
                    - keyword3
                    ...

                    # Categories
                    - category1
                    - category2

                    # Classification
                    - [Commercial/Editorial]: Provide the classification and a one-sentence justification based on the criteria provided.

                    # Mature Content
                    - [Yes/No]: Indicate if the image contains mature content and why.

                    # Illustration
                    - [Yes/No]: Indicate if the image qualifies as an illustration and why.
                    """
                )

            data = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": prompt_to_use,
                    "images": [image_path]
                }],
                #  "raw": "true",
                #  "format": "json",
                # "stream": "false",
                "options": {
                    "temperature": 0.5,
                    "top_p": 1.0
                }
            }

            if advanced_options:
                data["options"].update(advanced_options.get("options", {}))

            print("Sending request to the model...")
            print("Data sent:", json.dumps(data, indent=4))  # Debug: Show request payload

            response = self.client.chat(
                model=self.model,
                messages=data["messages"],
                options=data["options"]
            )

            # Output the raw response for inspection
            if response:
                print("Raw response from the model:", response)  # Show raw response
                return response.message.content.strip()
            else:
                return "An error occurred: No response received from the model."

        except Exception as e:
            return f"An error occurred: {e}"

    @staticmethod
    def parse_raw_response(raw_response):
        """
        Parse the raw response into a structured dictionary.

        Args:
            raw_response (str): Raw response from the model.

        Returns:
            dict: Parsed description, keywords, and categories.
        """
        try:
            description_match = re.search(
                r"(?:\*\*|# )Description(?:\*\*|)\s*\n(.+?)(?=\n\s*(?:\*\*|# )Keywords|$)",
                raw_response,
                re.S
            )
            description = description_match.group(1).strip() if description_match else ""

            keywords_match = re.search(r"(?:\*\*|# )Keywords(?:\*\*|)\s*((?:.+?\n)+?)(?=\n#|\*\*Categories|$)",
                                       raw_response, re.S)

            categories_match = re.search(
                r"(?:\*\*|# )Categories(?:\*\*|)\s*(.+?)(?=\n\s*(?:\*\*|# )Classification|$)",
                raw_response,
                re.S
            )

            classification_match = re.search(
                r"(?:\*\*|# )Classification(?:\*\*|):?\s*(?:-\s*)?(?:\*\*(Commercial|Editorial)\*\*|(?:\[?(Commercial|Editorial)\]?|(?:Commercial|Editorial)(?:\s*[-:].*)))",
                raw_response
            )
            mature_content_match = re.search(r"(?:\*\*|# )?Mature Content(?:\*\*|):?\s*(?:-\s*)?\[?(Yes|No)\]?:?.*",
                                             raw_response,
                                             re.S)
            illustration_match = re.search(r"(?:\*\*|# )?Illustration(?:\*\*|):?\s*(?:-\s*)?\[?(Yes|No)\]?:?.*",
                                           raw_response, re.S)

            keywords = (
                [kw.strip("-*• ").strip() for kw in keywords_match.group(1).splitlines() if kw.strip()]
                if keywords_match else []
            )
            categories = (
                [cat.strip(":-*• ").strip() for cat in categories_match.group(1).splitlines() if cat.strip()]
                if categories_match else []
            )

            classification = classification_match.group(1) or classification_match.group(
                2) if classification_match else None

            editorial = "yes" if classification and classification.lower() == "editorial" else "no"

            mature_content = mature_content_match.group(1).strip().lower() if mature_content_match else "no"
            illustration = illustration_match.group(1).strip().lower() if illustration_match else "no"

            return {
                "description": description,
                "keywords": keywords,
                "categories": categories,
                "editorial": editorial,
                "mature_content": mature_content,
                "illustration": illustration,
            }

        except Exception as e:
            print(f"Error parsing response: {e}")
            return {
                "description": "",
                "keywords": [],
                "categories": [],
                "editorial": "no",
                "mature_content": "No",
                "illustration": "No",
            }

    @staticmethod
    def save_to_csv(results, image_path, file_path):
        """
        Save analysis results to a CSV file.

        Args:
            results (dict): A dictionary containing "description", "keywords", and "categories".
            image_path (str): Name or path of the image file.
            file_path (str): Path to the CSV file.
            options (dict): Additional options for other columns (e.g., "Editorial", "Mature content", "Illustration").
        """
        # Prepare data to append
        row = {
            "Filename": image_path.strip(),
            "Description": results.get("description", "").strip(),
            "Keywords": ", ".join(results.get("keywords", [])).strip(),
            "Categories": ", ".join(results.get("categories", [])).strip(),
            "Editorial": results.get("editorial", "no").strip(),
            "Mature content": results.get("mature_content", "no").strip(),
            "Illustration": results.get("illustration", "no").strip(),
        }

        # Ensure correct column order
        column_order = ["Filename", "Description", "Keywords", "Categories", "Editorial",
                        "Mature content", "Illustration"]

        # Check if file exists
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Reorder columns to match the expected order
                for col in column_order:
                    if col not in df.columns:
                        df[col] = ""  # Add missing columns with empty values
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

    custom_prompt = (
        """
        Analyze this image and provide the following details:
        """
    )

    # Optional advanced options
    advanced_options = {
        "options": {
            "temperature": 0.5,
            "top_p": 1.0
        }
    }

    # Analyze the image
    # result = analyzer.analyze_image(image_path, custom_prompt, advanced_options=advanced_options)
    result = analyzer.analyze_image(image_path)

    print("Raw Response:\n", result)

    # Parse the raw response
    parsed_result = analyzer.parse_raw_response(result)

    # Save to CSV
    analyzer.save_to_csv(parsed_result, image_path, file_path)


