# Image Analyzer Project

This project leverages **Ollama** and AI models to analyze images and prepare metadata suitable for Shutterstock. The tool processes images to generate:

1.  A descriptive text for the image.
2.  A list of keywords relevant to the image content.
3.  Appropriate categories for classification.

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
git clone https://github.com/ShutterstockimageAnalyzer/image-analyzer.git
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