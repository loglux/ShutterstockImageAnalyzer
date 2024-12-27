# Shutterstock Metadata Generator Powered by AI

This project utilizes Ollama and AI models to analyze images and automatically generate metadata tailored for Shutterstock. By automating the creation of descriptive text, keywords, and categories, the tool streamlines the process of preparing images for upload, saving time and reducing manual effort.

The tool simplifies contributing high-quality images to Shutterstock or similar platforms by providing:

1. A descriptive text summarizing the image.
2. A list of keywords relevant to the image content.
3. Suggested categories for classification.

This automation is particularly beneficial for photographers, content creators, and agencies looking to enhance their workflow and focus more on creativity rather than metadata management.

## Features

-   Uses the `llama3.2-vision` model hosted locally.
-   Parses AI responses to maintain consistent and clean metadata formatting.
-   Outputs results in Shutterstock-compatible CSV format.
-   Configurable options for advanced analysis.

## Prerequisites

-   **Ollama** installed and configured locally with GPU support.
-   Python 3.10 or later.
-   The following Python packages:
    -   `json`
    -   `pandas`
    -   `os`
    -   `re`

## Installation

1.  Clone the repository:
```bash
git clone https://github.com/loglux/ShutterstockImageAnalyzer.git
cd image-analyzer
```
2.  Install dependencies:
```bash
pip install pandas
``` 
    
3.  Ensure **Ollama** is running on the local host.
   

## Usage

1.  Update the `base_url` in the script to match your Ollama instance:
    
```python
base_url="http://localhost:11434/"
``` 
    
2.  Run the script with your image file:
```bash
python image_analyzer.py
``` 
    
3.  The results will be saved to `shutterstock.csv` in the project directory.
    

## File Details

-   `image_analyzer.py`: Core script for image analysis and metadata extraction.
-   `shutterstock.csv`: Output CSV file containing image metadata.