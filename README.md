# Rufus: Intelligent Web Data Extraction Tool

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Requirements](#requirements)
6. [Configuration](#configuration)
7. [How It Works](#how-it-works)
8. [Error Handling](#error-handling)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview
Rufus is an intelligent web data extraction tool designed to crawl websites and extract relevant information based on user-defined instructions. Utilizing advanced language models and web scraping techniques, Rufus aims to provide structured and meaningful data for various applications, including research, data analysis, and content generation.

## Features
- **Web Crawling**: Automatically navigates through web pages to gather information.
- **Data Extraction**: Extracts relevant content based on user instructions.
- **LLM-Based Relevance Checking**: Uses a language model to determine the relevance of content before processing.
- **Duplicate Content Filtering**: Identifies and skips duplicate information to ensure data quality.
- **Configurable Parameters**: Users can set maximum pages to crawl, depth of links to follow, and more.
- **Structured Output**: Results are saved in a consistent JSON format for easy consumption.

## Installation
To set up Rufus, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/rufus.git
   cd rufus
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Ensure you have `pip` installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run Rufus, use the following command in your terminal:
```bash
python rufus_v4.py --url <URL> --instructions "<Your instructions here>"
```

### Example
```bash
python rufus_v4.py --url https://www.bamboohr.com/resources/hr-glossary/california-labor-laws --instructions "Extract information about FEHA"
```

## Requirements
Rufus requires the following Python libraries:
- `agno`
- `beautifulsoup4`
- `requests`
- `selenium`
- `webdriver-manager`
- `python-dotenv`
- `groq`
- `duckduckgo-search`

These libraries are specified in the `requirements.txt` file.

## Configuration
You can configure Rufus using command-line arguments:
- `--url`: The starting URL for crawling.
- `--instructions`: Instructions for data extraction.
- `--max_pages`: Maximum number of pages to crawl (default: 3).
- `--max_depth`: Maximum depth of links to follow (default: 1).
- `--output`: Name of the output file (default: `rufus_output.json`).
- `--log_file`: Path to the log file.
- `--verbose`: Enable verbose logging.
- `--similarity`: Similarity threshold for duplicate detection (default: 0.7).
- `--save-intermediate`: Save intermediate results after each page.
- `--disable-llm-relevance`: Disable LLM-based relevance checking.

## How It Works
1. **Crawling**: Rufus starts at the specified URL and extracts links to follow.
2. **Data Extraction**: For each page, it checks relevance using the LLM and extracts structured data.
3. **Output**: The extracted data is saved in a JSON format, which includes metadata such as source URL and extraction time.

## Error Handling
Rufus includes robust error handling to manage issues such as:
- Invalid URLs
- Network errors during scraping
- Content parsing errors

In case of an error, Rufus saves any collected data to an error file for review.

## Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
