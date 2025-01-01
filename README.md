# **Shutterstock Metadata Generator Powered by AI**

This project uses **Ollama** and advanced AI models to automate metadata generation for Shutterstock images. By analyzing images and extracting meaningful details, it streamlines the process of creating descriptive text, keywords, and categories, saving time and reducing manual effort.

This tool is ideal for photographers, content creators, and agencies aiming to focus more on creativity and less on tedious metadata tasks.

### **What It Does**

1.  **Description**: Generates a descriptive text summarizing the image.
2.  **Keywords**: Produces a list of relevant keywords.
3.  **Categories**: Suggests appropriate categories for the image.
4.  **Image Classification**: Identifies if the image is commercial or editorial.
5.  **Mature Content Detection**: Flags images with nudity, violence, or unsuitable content.
6.  **Illustration Detection**: Recognizes digitally created or heavily manipulated images.

----------

## **Features**

-   Utilizes the `llama3.2-vision` model hosted locally.
-   Parses AI responses to provide clean and consistent metadata.
-   Outputs results in Shutterstock-compatible CSV format.
-   Supports customizable prompts and advanced analysis options.
-   Recursive directory processing for batch operations.

----------

## **Prerequisites**

-   **Ollama** installed and configured locally with GPU support.
    -   Recommended hardware: Modern NVIDIA RTX series GPU.
    -   Tested and works flawlessly on **NVIDIA RTX 4070**.
-   **Python 3.10+**.
-   Required Python libraries:
    -   `ollama`
    -   `pandas`
    -   `pydantic`

----------

## **Installation**

1.  Clone the repository:
    
```bash
git clone https://github.com/loglux/ShutterstockImageAnalyzer.git
cd ShutterstockImageAnalyzer` 
```
   
2.  Install dependencies:
    
 ```bash
pip install ollama pandas pydantic
 ``` 
    
3.  Ensure **Ollama** is running on your local machine:
    
```bash
ollama run llama3.2-vision
``` 
   
----------

## **Usage**

### **1. Configuration**

Update `base_url` to match your Ollama instance:
```python
base_url = "http://localhost:11434/"
``` 

Set the paths for the image directory and output file:
```python
image_directory_path = r"ShutterstockImageAnalyzer"  
csv_file_path = r"shutterstock.csv"
```
Enable or disable recursive processing:
```python
recursive=True  # Process subfolders
recursive=False  # Only process files in the main directory
``` 

### **2. Run the Script**
Execute the script to process images:
```bash
python image_analyzer.py`
```

### **3. View the Results**

The metadata will be saved to `shutterstock.csv` in the project directory.

### **4. Customize Prompts and Options**

Modify the prompt or advanced options to fine-tune model behavior:

-   **Custom Prompt**:
```python
prompt = "Describe the image, highlighting its objects and their relationships."
``` 
    
-   **Advanced Options**:
```python
advanced_options = {"temperature": 0.7, "top_p": 0.95}
``` 
   

### **5. Process a Directory**

Analyze all images in a directory:
```python
analyzer.process_images_in_directory(
    directory_path="path/to/images",
    file_path="results.csv",
    prompt="Analyze this image and provide details...",
    advanced_options={"temperature": 0.6},
    recursive=True  # Enable subfolder processing
)
```

----------

## **Default Behaviors**

-   **Recursive Directory Search**: Enabled by default.
-   **Prompt and Options**: Defaults to a generic prompt and balanced settings unless specified.

### **Custom Execution**

To analyze a single image, modify the script’s `__main__` block or use the following method:

```python
analyzer.start_analysis(
    image_path="path/to/image.jpg",
    file_path="results.csv",
    prompt="Analyze this image and generate metadata."
)
``` 
----------

## **File Details**

-   **`image_analyzer.py`**: Core script for metadata generation and processing.
-   **`shutterstock.csv`**: Output file containing structured metadata.
----------
## Prompt Compliance Evaluation
The evaluation method provides a summary of how well the generated results adhere to the prompt's requirements. Please note:
- **Compliance may not always be 100%** due to factors like repetitive starting phrases, insufficient keyword counts, or model-specific constraints.
- The provided statistics are intended to guide improvements in model settings or prompt design.

### Example Usage
```python
compliance_stats = analyzer.evaluate_prompt_compliance("results.csv")
print(compliance_stats)
```
### What the Metrics Represent:
The evaluation method provides detailed metrics to assess how well the generated results adhere to the prompt's requirements. Below is a breakdown of each metric:

- **`description_compliance`**:
  - Percentage of descriptions with a length not exceeding 200 characters.
  - **Why it matters**: Ensures descriptions are concise and meet platform constraints.

- **`keyword_min_compliance`**:
  - Percentage of cases where at least 7 keywords are generated.
  - **Why it matters**: Guarantees a minimum level of diversity and relevance in keywords.

- **`keyword_max_compliance`**:
  - Percentage of cases where the number of keywords does not exceed 50.
  - **Why it matters**: Prevents overly verbose outputs while keeping the keywords manageable.

- **`category_compliance`**:
  - Percentage of cases where exactly 1 or 2 categories are returned.
  - **Why it matters**: Ensures strict adherence to the requirement of selecting categories from a predefined list without exceeding the allowed number.

- **`description_uniqueness`**:
  - Percentage of unique descriptions. The higher, the better.
  - **Why it matters**: Encourages variety and creativity in generated outputs, avoiding repetition across multiple results.

- **`duplicate_start_phrases`**:
  - Percentage of descriptions starting with the same first 5 words.
  - **Why it matters**: A high percentage might indicate formulaic or repetitive outputs, but this metric is less critical for some use cases. Depending on the application, repetitive starting phrases may or may not impact the overall quality of the results.

### Key Notes:
1. **Focus on the critical metrics**:
   - Metrics like `description_compliance`, `keyword_min_compliance`, `keyword_max_compliance`, and `category_compliance` are generally more important for ensuring adherence to platform and prompt requirements.
   
2. **Secondary metrics (`description_uniqueness`, `duplicate_start_phrases`)**:
   - While these provide insights into the variety and diversity of outputs, they may not be crucial for all applications.
   - For instance, repetitive starting phrases (`duplicate_start_phrases`) might not matter as long as the descriptions are otherwise unique and meet length requirements.

3. **Using the metrics effectively**:
   - Use the compliance metrics as a diagnostic tool to identify areas for improvement in the prompt or model settings.
   - Treat lower scores on `description_uniqueness` or `duplicate_start_phrases` as a potential improvement area, rather than a strict requirement.

----------
### **Alternative Models**

While the tool is optimized for **`llama3.2-vision`**, it also supports other vision models, such as **`llava-7b`** and **`llava-llama3-8b`**, which come with their own advantages and limitations:

#### **Model Options**

1.  **Llama3.2-vision**:
   
    -   **Strengths**: Highly accurate, capable of recognizing well-known objects (e.g., "Eiffel Tower" instead of "tower" or "landmark").
    -   **Weaknesses**: Slightly slower compared to other models.
2.  **Llava**:
   
    -   **Strengths**: Faster than `llama3.2-vision`, suitable for larger datasets.
    -   **Weaknesses**: Limited to general concepts like "bird" or "monkey" without specifying exact species or object types.
3.  **Llava-llama3**:
    
    -   **Strengths**: Extremely fast, capable of creating more emotional and engaging descriptions.
    -   **Weaknesses**: May not follow specific instructions (e.g., generating at least 7 keywords) and struggles with detailed object recognition.

----------

### **Customizing for Other Models**

When using alternative models, some parameters or parts of the prompt may need to be adapted. For example:

-   **Keywords**: Models like `llava` may not generate a specific number of keywords as instructed. You might need to post-process or fine-tune prompts to address this.
-   **Descriptions**: While less precise in identifying objects, `llava` models excel at creating emotional and atmospheric descriptions.
-   **Categories**: Consider simplifying the list of categories, as some models may perform better with fewer options.

To switch models, update the `model` parameter in the script:

```python
analyzer = ImageAnalyzer(model="llava", base_url="http://localhost:11434/")
``` 

----------

### **Choosing the Right Model**

If speed is a priority (e.g., for processing thousands of images), models like `llava` or `llava-llama3` might be more suitable. However, if accuracy in recognizing real-world objects and providing detailed classifications (e.g., identifying specific landmarks, species, or distinct features) is essential, `llama3.2-vision` remains the best choice.