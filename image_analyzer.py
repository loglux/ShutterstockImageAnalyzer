import time
from ollama import Client
from pydantic import BaseModel, ValidationError
import pandas as pd
import os
from pathlib import Path
from typing import Iterable, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from PIL import Image
except ImportError:
    Image = None

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

    def analyze_image(self, image_path, prompt=None, advanced_options=None, hint=None, max_retries=3):
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                # Define default prompt if none is provided
                if not prompt:
                    prompt = (
                        """
                        Analyze this image and provide the following details:
1. Provide an engaging image caption. The caption must be under 200 characters. Do not exceed this limit under any circumstances. Use one concise sentence, no filler words like "serene/picturesque", no ellipsis, no repetition of adjectives. Avoid assumptions or guesses. If possible, specify the exact name of the object, landmark, or location (e.g., "Eiffel Tower" instead of "tower" or "landmark"; "Bald Eagle" instead of "bird"). Use specific terms for identifiable features visible in the image, avoiding overly generic descriptions.
2. Generate no fewer than 7 and up to 50 unique and relevant keywords describing the image.  
    - Keywords must be UNIQUE, max 30 items after deduplication.  
    - No placeholders, no “…”, no slashes, no HTML/Markdown.  
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

Return ONLY a single JSON object, no extra text, no markdown, no code fences, no ellipsis. If a field is unknown, use an empty string or false instead of freeform text.

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
                        "num_predict": 600,  # low value causes JSON errors
                        "top_k": 260,  # should increase the diversity of keywords.
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
                result = ImageAnalysisResult.model_validate_json(response.message.content)

                # Validating the length of the description:
                if len(result.description) > 200:
                    print("Warning: Generated caption exceeds 200 characters. Attempting to rewrite.")
                    # prompt = f"Please shorten the following image caption to fit within 200 characters while retaining its meaning and key details:\n\n{result.description}"
                    prompt = f"Rewrite the following image caption to be concise and fit within 200 characters. Provide only the revised caption without any explanations:\n\n{result.description}"
                    rewrite_response = self.client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result.description = rewrite_response.message.content.strip()

                return result


            except ValidationError as e:
                print(f"Validation error occurred on attempt {attempt}: {e}")
                if "json_invalid" in str(e):
                    print("JSON error, retrying...")
                    time.sleep(1) # a little pause before new attempt
                    continue
                else:
                    # if the error is not 'json_invalid'
                    print(f"Error: {e}. Skipping this image.")
                    return f"Error: {e}. Skipping this image."

        print(f"Failed to analyze image after {max_retries} attempts: {image_path}")
        return f"Failed to analyze image after {max_retries} attempts: {image_path}"


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
        cleaned_keywords = self.clean_keywords(results.keywords)
        # Prepare data to append
        row = {
            # "Filename": image_path.strip(),
            "Filename": os.path.basename(image_path.strip()),
            "Description": results.description.strip(),
            "Keywords": ", ".join(cleaned_keywords).strip(),
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

    def start_analysis(self, image_path, file_path, prompt=None, advanced_options=None, hint=None, max_retries=3, resize_max=None, resize_quality=85):
        """
        Analyze an image and save the results to a CSV file.
        Args:
        :param image_path (str): Path to the image file.
        :param file_path (str): Path to the CSV file.
        :param promp (str, optional): Prompt to use for analysis. Defaults to None.
        :param advanced_options (dict, optional): Advanced options for the analysis. Defaults to None.
        :param max_retries (int, optional): Attempts allowed when the model returns invalid JSON. Defaults to 4.
        :param resize_max (int, optional): Max dimension to resize images before sending. Defaults to None.
        :param resize_quality (int, optional): JPEG quality for resized images. Defaults to 85.
        """
        prepared_image_path, cleanup_cb = self.prepare_image(image_path, resize_max, resize_quality)
        # Analyze the image
        result = self.analyze_image(prepared_image_path, prompt, advanced_options, hint=hint, max_retries=max_retries)

        # Ensure result is structured before proceeding
        if isinstance(result, ImageAnalysisResult):
            print("Analysis Result:", result)
            self.save_to_csv(result, image_path, file_path)
            return True

        print("Failed to analyze image:", result)
        return False

    def process_images_in_directory(self, directory_path, file_path, prompt=None, advanced_options=None, recursive=True, hint=None, max_retries=3, show_progress=True, enable_fallback=True, fallback_max_retries=3, fallback_overrides=None, resize_max=None, resize_quality=85):
        """
        Search and process all images in a directory and subdirectories.
        Args:
            directory_path (str): Path to the directory containing images.
            file_path (str): Path to the CSV file to save results.
            prompt (str, optional): Prompt to use for analysis. Defaults to None.
            advanced_options (dict, optional): Advanced options for the analysis. Defaults to None.
            recursive (bool, optional): Search recursively in subdirectories. Defaults to True.
            max_retries (int, optional): Attempts allowed when the model returns invalid JSON. Defaults to 4.
            show_progress (bool, optional): Display progress bar or per-file progress. Defaults to True.
            enable_fallback (bool, optional): Re-run failed images with fallback options. Defaults to True.
            fallback_max_retries (int, optional): Attempts in fallback phase. Defaults to 3.
            fallback_overrides (dict, optional): Option overrides for fallback phase.
            resize_max (int, optional): Max dimension to resize images before sending. Defaults to None.
            resize_quality (int, optional): JPEG quality for resized images. Defaults to 85.
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

        total_images = len(image_files)
        if not total_images:
            print(f"No images found in directory: {directory_path}")
            return

        print(f"Found {total_images} images in directory: {directory_path}")

        use_tqdm = show_progress and tqdm is not None
        iterator: Iterable[Tuple[int, Path]]
        if use_tqdm:
            progress = tqdm(image_files, total=total_images, unit="img", dynamic_ncols=True)
            iterator = enumerate(progress, 1)
        else:
            iterator = enumerate(image_files, 1)

        # Processing images one by one
        failed_images = []
        for index, image_path in iterator:
            if use_tqdm:
                progress.set_description(f"Processing {image_path.name}")
            elif show_progress:
                print(f"[{index}/{total_images}] Processing: {image_path}")
            success = self.start_analysis(str(image_path), file_path, prompt, advanced_options, hint=hint, max_retries=max_retries, resize_max=resize_max, resize_quality=resize_quality)
            if not success:
                failed_images.append(str(image_path))
            time.sleep(0.5)

        # Fallback pass for failed images with safer options
        if enable_fallback and failed_images:
            print(f"Retrying {len(failed_images)} failed images with fallback options...")
            overrides = fallback_overrides or {
                "temperature": 0.35,
                "top_k": 120,
                "num_predict": 600,
            }
            # merge original advanced_options with fallback overrides
            def merge_options(base_opts, override_opts):
                merged = {"options": {}}
                if base_opts:
                    merged["options"].update(base_opts.get("options", {}))
                merged["options"].update(override_opts)
                return merged

            fallback_opts = merge_options(advanced_options, overrides)

            for image_path in failed_images:
                print(f"Fallback processing: {image_path}")
                self.start_analysis(image_path, file_path, prompt, fallback_opts, hint=hint, max_retries=fallback_max_retries, resize_max=resize_max, resize_quality=resize_quality)
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
    def clean_keywords(keywords, max_keywords=50):
        """
        Deduplicate and trim keyword list, dropping empty/short tokens and limiting length.
        """
        cleaned = []
        seen = set()
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            token = kw.replace("\xa0", " ").strip()
            if not token:
                continue
            if len(token) <= 1:
                continue
            if all(ch in {".", ","} for ch in token):
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(token)
            if len(cleaned) >= max_keywords:
                break
        return cleaned

    def prepare_image(self, image_path, resize_max=None, resize_quality=85):
        """
        Optionally resize an image before sending to the model.
        Returns the path to use and an optional cleanup callback (unused when caching).
        """
        if not resize_max or not Image:
            return image_path, None

        try:
            cache_dir = Path(".cache/resized")
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_path = cache_dir / f"{Path(image_path).stem}_max{resize_max}_q{resize_quality}.jpg"

            if output_path.exists() and output_path.stat().st_mtime >= Path(image_path).stat().st_mtime:
                return str(output_path), None

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((resize_max, resize_max))
                img.save(output_path, format="JPEG", quality=resize_quality, optimize=True)

            return str(output_path), None
        except Exception as exc:
            print(f"Resize failed for {image_path}, using original: {exc}")
            return image_path, None

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
    import argparse
    import sys

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Analyze images with an Ollama vision model and write Shutterstock-compatible metadata to CSV."
        )
        parser.add_argument("--dir", default=".", help="Directory with images to process (ignored if --image is set).")
        parser.add_argument("--image", help="Process a single image file instead of a directory.")
        parser.add_argument("--csv", default="shutterstock.csv", help="Path to the output CSV file.")
        parser.add_argument(
            "--model",
            default=os.getenv("OLLAMA_MODEL", "llama3.2-vision"),
            help="Model name to use (defaults to OLLAMA_MODEL env or 'llama3.2-vision').",
        )
        parser.add_argument(
            "--base-url",
            dest="base_url",
            default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"),
            help="Ollama base URL (defaults to OLLAMA_BASE_URL env or http://localhost:11434/).",
        )
        parser.add_argument(
            "--recursive",
            action="store_true",
            help="Recurse into subdirectories when processing a directory (default: off).",
        )
        parser.add_argument("--hint", help="Optional hint prepended to the prompt.")
        parser.add_argument(
            "--max-retries",
            type=int,
            default=3,
            help="Max attempts when the model returns invalid JSON before fallback (default: 3).",
        )
        parser.add_argument(
            "--resize-max",
            type=int,
            help="Resize images to this max dimension (pixels) before sending to the model. Requires Pillow.",
        )
        parser.add_argument(
            "--resize-quality",
            type=int,
            default=85,
            help="JPEG quality for resized images (default: 85).",
        )
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable progress output (tqdm if installed, otherwise simple counters).",
        )
        parser.add_argument("--prompt-file", dest="prompt_file", help="Path to a file containing a custom prompt.")
        parser.add_argument("--temperature", type=float, help="Override temperature option.")
        parser.add_argument("--top-p", dest="top_p", type=float, help="Override top_p option.")
        parser.add_argument("--top-k", dest="top_k", type=int, help="Override top_k option.")
        parser.add_argument("--num-predict", dest="num_predict", type=int, help="Override num_predict option.")
        parser.add_argument("--num-ctx", dest="num_ctx", type=int, help="Override num_ctx option.")
        parser.add_argument(
            "--no-fallback",
            action="store_true",
            help="Disable fallback re-processing for files that fail initial attempts.",
        )
        return parser.parse_args()

    args = parse_args()

    custom_prompt = None
    if args.prompt_file:
        try:
            custom_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Error reading prompt file: {exc}")
            sys.exit(1)

    option_overrides = {}
    for opt_name in ("temperature", "top_p", "top_k", "num_predict", "num_ctx"):
        value = getattr(args, opt_name)
        if value is not None:
            option_overrides[opt_name] = value

    advanced_options = {"options": option_overrides} if option_overrides else None

    analyzer = ImageAnalyzer(model=args.model, base_url=args.base_url)

    if args.image:
        analyzer.start_analysis(
            args.image,
            args.csv,
            prompt=custom_prompt,
            advanced_options=advanced_options,
            hint=args.hint,
            max_retries=args.max_retries,
            resize_max=args.resize_max,
            resize_quality=args.resize_quality,
        )
    else:
        analyzer.process_images_in_directory(
            args.dir,
            args.csv,
            prompt=custom_prompt,
            advanced_options=advanced_options,
            recursive=args.recursive,
            hint=args.hint,
            max_retries=args.max_retries,
            show_progress=not args.no_progress,
            enable_fallback=not args.no_fallback,
            resize_max=args.resize_max,
            resize_quality=args.resize_quality,
        )
