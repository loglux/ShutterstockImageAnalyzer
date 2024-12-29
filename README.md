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