from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os
import json
import time
import argparse
import logging
import sys
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import warnings
from difflib import SequenceMatcher


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Configure custom logger
class RufusLogger:
    def __init__(self, log_level=logging.INFO, log_file=None):
        self.logger = logging.getLogger("rufus")
        self.logger.setLevel(log_level)
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def critical(self, msg):
        self.logger.critical(msg)

class Rufus:
    def __init__(self, max_pages=10, max_depth=2, log_level=logging.INFO, log_file=None, 
                 similarity_threshold=0.7, save_intermediate=False, llm_relevance_check=True):
        """
        Initialize Rufus web crawler and data extractor
        
        Args:
            max_pages (int): Maximum number of pages to crawl
            max_depth (int): Maximum depth of links to follow
            log_level (int): Logging level
            log_file (str): Path to log file
            similarity_threshold (float): Threshold for duplicate detection
            save_intermediate (bool): Whether to save intermediate results
            llm_relevance_check (bool): Whether to use LLM for relevance checking
        """
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.visited_urls = set()
        self.to_visit = []
        self.extracted_data = []
        self.similarity_threshold = similarity_threshold
        self.save_intermediate = save_intermediate
        self.llm_relevance_check = llm_relevance_check
        
        self.logger = RufusLogger(log_level=log_level, log_file=log_file)
        self.logger.info("Initializing Rufus...")
        
        self.driver = None
        self._init_selenium()
        
        self._init_agent()
    
    def _init_selenium(self):
        """Initialize Selenium WebDriver with robust error handling"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-logging")
            chrome_options.add_argument("--log-level=3")  # Only fatal errors
            chrome_options.add_argument("--silent")
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            with open(os.devnull, 'w') as f:
                webdriver_service = Service(
                    ChromeDriverManager().install()
                )
            
            self.driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
            raise RuntimeError(f"Failed to initialize Selenium: {e}")
    
    def _init_agent(self):
        """Initialize Agno agent with error handling"""
        try:
            self.agent = Agent(
                name="Rufus",
                model=Groq(id="llama-3.3-70b-versatile"),
                tools=[DuckDuckGoTools()],
                description="You are Rufus, an intelligent web data extraction tool.",
                instructions=[
                    "Extract relevant information from the provided content based on the user's instructions.",
                    "If needed, search the web for additional information using the provided tools.",
                    "After gathering all necessary information, provide a final response that does NOT include function calls.",
                    "Focus only on content that matches the instructions.",
                    "Structure your final response in a consistent JSON format with these fields:",
                    "- 'title': A concise title for the extracted information",
                    "- 'content': The main extracted information, organized in a structured way",
                    "- 'metadata': Additional information including source URL and date",
                    "Make sure your final response is clean, well-formatted JSON without any function calls or tool usage text."
                ],
                show_tool_calls=True,
                markdown=True
            )
            self.logger.info("Agno agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Agno agent: {e}")
            raise RuntimeError(f"Failed to initialize Agno agent: {e}")
    
    def is_valid_url(self, url):
        """Check if a URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            self.logger.warning(f"Invalid URL format: {url}, Error: {e}")
            return False
    
    def scrape_page(self, url):
        """
        Scrape a single page and extract its content and links
        
        Args:
            url (str): URL to scrape
            
        Returns:
            tuple: (content, links)
        """
        self.logger.info(f"Scraping {url}...")
        
        if not self.driver:
            self.logger.error("WebDriver not initialized")
            return None, []
        
        try:
            self.driver.set_page_load_timeout(30)
            
            self.driver.get(url)
            
            time.sleep(3)
            
            page_source = self.driver.page_source
            
            soup = BeautifulSoup(page_source, 'html.parser')
            
            text_content = soup.get_text(separator=' ', strip=True)
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                link = a_tag['href']
                if link and not link.startswith('#') and not link.startswith('javascript:'):
                    links.append(link)
            
            self.logger.info(f"Found {len(links)} links on {url}")
            return text_content, links
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            
            try:
                self.logger.info(f"Trying fallback scraping method for {url}")
                self.driver.get(url)
                text_content = self.driver.page_source
                return text_content, []
            except:
                return None, []
    
    def is_relevant(self, content, instructions, min_length=100):
        """
        Check if the page content is relevant to the user's instructions
        
        Args:
            content (str): Page content
            instructions (str): User instructions
            min_length (int): Minimum content length to be considered
            
        Returns:
            bool: True if relevant, False otherwise
        """
        
        if not content or len(content) < min_length:
            return False
        
        if not self.llm_relevance_check:
            return True
        
        content_sample = content[:5000]  
        
        try:
            prompt = f"""
            I need to determine if the following content is relevant to this instruction: "{instructions}"
            
            Here's a sample of the content:
            ---
            {content_sample}
            ---
            
            Please respond with ONLY "YES" if the content is relevant to the instruction, or "NO" if it's not relevant.
            Do not include any other text in your response.
            """
            
            response = self.agent.run(prompt)
            
            response_text = response.content.strip().upper()
            
            is_relevant = "YES" in response_text
            
            self.logger.info(f"Relevance check: {'Relevant' if is_relevant else 'Not relevant'}")
            return is_relevant
            
        except Exception as e:
            self.logger.warning(f"Error checking relevance: {e}")
            return True
    
    def extract_data(self, url, content, instructions):
        """
        Extract structured data from page content based on user instructions
        
        Args:
            url (str): Page URL
            content (str): Page content
            instructions (str): User instructions
            
        Returns:
            dict: Structured data
        """

        if not content or len(content) < 50:
            self.logger.warning(f"Content too short or empty for {url}")
            return {
                "url": url,
                "extracted_content": "No meaningful content found on this page.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        url_extension = os.path.splitext(url)[1].lower()
        if url_extension in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.mp4', '.webm', '.mp3', '.wav']:
            self.logger.info(f"Skipping non-text content: {url}")
            return {
                "url": url,
                "extracted_content": "No meaningful content found on this page.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        try:
            prompt = f"""
            I've scraped the following webpage: {url}
            
            Here's the content:
            {content[:5000]}... [content truncated for brevity]
            
            Based on these instructions: "{instructions}", extract the relevant information.
            Format your response as a clean JSON object with 'title', 'content', and 'metadata' fields.
            Do not include any function calls or tool usage text in your final response.
            """
            
            self.logger.info(f"Extracting data from {url} with instructions: {instructions}")
            response = self.agent.run(prompt)
            
            if response and response.content:
                
                extracted_content = self._parse_json_from_response(response.content, url)
                
                
                result = {
                    "url": url,
                    "extracted_content": extracted_content,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return result
            else:
                self.logger.warning(f"No response from agent for {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting data from {url}: {e}")
            return None
    
    def _clean_function_calls(self, text):
        """Remove function call syntax from text"""
        if not isinstance(text, str):
            return text
        
        cleaned = re.sub(r'<function=.*?</function>', '', text)
        cleaned = re.sub(r'Running:.*?\n', '', cleaned)
        
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        return cleaned.strip()
    
    def _parse_json_from_response(self, response_text, url):
        """
        Parse JSON from the agent's response
        
        Args:
            response_text (str): Agent's response text
            url (str): Source URL
            
        Returns:
            dict: Parsed JSON or structured content
        """
        cleaned_text = self._clean_function_calls(response_text)
        
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_matches = re.findall(json_pattern, cleaned_text)
        
        if json_matches:
            try:
                json_content = json.loads(json_matches[0])
                
                if not isinstance(json_content, dict):
                    json_content = {"content": json_content}
                
                if "title" not in json_content:
                    json_content["title"] = "Extracted Information"
                
                if "metadata" not in json_content:
                    json_content["metadata"] = {}
                
                if "source" not in json_content["metadata"]:
                    json_content["metadata"]["source"] = url
                
                if "extraction_time" not in json_content["metadata"]:
                    json_content["metadata"]["extraction_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                return json_content
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON from response for {url}")
        
        try:
            json_content = json.loads(cleaned_text)
            

            if not isinstance(json_content, dict):
                json_content = {"content": json_content}
            
            if "title" not in json_content:
                json_content["title"] = "Extracted Information"
            
            if "metadata" not in json_content:
                json_content["metadata"] = {}
            
            if "source" not in json_content["metadata"]:
                json_content["metadata"]["source"] = url
            
            if "extraction_time" not in json_content["metadata"]:
                json_content["metadata"]["extraction_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return json_content
        except json.JSONDecodeError:
            pass
        
        lines = cleaned_text.split('\n')
        title = "Extracted Information"
        content = cleaned_text
        
        if lines and lines[0].strip():
            first_line = lines[0].strip()
            if first_line.startswith('#'):
                title = first_line.lstrip('#').strip()
                content = '\n'.join(lines[1:]).strip()
            elif len(first_line) < 100:
                title = first_line
                content = '\n'.join(lines[1:]).strip()
        
        return {
            "title": title,
            "content": content,
            "metadata": {
                "source": url,
                "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def _post_process_extracted_data(self):
        """Clean up and standardize the extracted data"""
        processed_data = []
        
        for item in self.extracted_data:
            if 'extracted_content' in item:
                content = item['extracted_content']
                
                if isinstance(content, str):
                    if content == "No meaningful content found on this page.":
                        processed_data.append(item)
                        continue
                    
                    try:
                        parsed = json.loads(content)
                        item['extracted_content'] = parsed
                    except:
                        cleaned = self._clean_function_calls(content)
                        item['extracted_content'] = {
                            "title": "Extracted Information",
                            "content": cleaned,
                            "metadata": {
                                "source": item.get("url", "unknown"),
                                "extraction_time": item.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                            }
                        }
                
                elif isinstance(content, dict):
                    if 'title' in content:
                        content['title'] = self._clean_function_calls(content['title'])
                    
                    if 'content' in content:
                        if isinstance(content['content'], str):
                            content['content'] = self._clean_function_calls(content['content'])
                        
                
                processed_data.append(item)
        
        return processed_data
    
    def _is_duplicate_content(self, new_content, existing_contents, threshold=None):
        """
        Check if the new content is a duplicate of existing content
        
        Args:
            new_content (str): New content to check
            existing_contents (list): List of existing content
            threshold (float): Similarity threshold (0.0 to 1.0)
            
        Returns:
            bool: True if duplicate, False otherwise
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        if not new_content or not existing_contents:
            return False
        
        if isinstance(new_content, dict) and 'content' in new_content:
            new_text = str(new_content['content'])
        else:
            new_text = str(new_content)
        
        for existing in existing_contents:
            if isinstance(existing, dict) and 'extracted_content' in existing:
                existing_content = existing['extracted_content']
                
                # Extract content field if it exists
                if isinstance(existing_content, dict) and 'content' in existing_content:
                    existing_text = str(existing_content['content'])
                else:
                    existing_text = str(existing_content)
                
                # Calculate similarity
                similarity = SequenceMatcher(None, new_text, existing_text).ratio()
                
                if similarity >= threshold:
                    self.logger.info(f"Duplicate content detected (similarity: {similarity:.2f})")
                    return True
        
        return False
    
    def crawl(self, start_url, instructions):
        """
        Crawl a website based on user instructions
        
        Args:
            start_url (str): Starting URL
            instructions (str): User instructions
            
        Returns:
            list: Extracted data from relevant pages
        """
        if not self.is_valid_url(start_url):
            self.logger.error(f"Invalid URL: {start_url}")
            return []
        
        
        domain = urlparse(start_url).netloc
        
        
        self.to_visit = [(start_url, 0)]  
        self.visited_urls = set()
        self.extracted_data = []
        
        self.logger.info(f"Starting crawl from {start_url} with instructions: {instructions}")
        self.logger.info(f"Maximum pages: {self.max_pages}, Maximum depth: {self.max_depth}")
        
        
        while self.to_visit and len(self.visited_urls) < self.max_pages:
            current_url, current_depth = self.to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            self.logger.info(f"Processing {current_url} (depth: {current_depth})")
            self.visited_urls.add(current_url)
            
            content, links = self.scrape_page(current_url)
            
            if not content:
                self.logger.warning(f"No content found at {current_url}")
                continue
            
            
            url_extension = os.path.splitext(current_url)[1].lower()
            if url_extension in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.mp4', '.webm', '.mp3', '.wav', '.pdf']:
                self.logger.info(f"Skipping non-text content: {current_url}")
                continue
            
            
            if not self.is_relevant(content, instructions):
                self.logger.info(f"Skipping irrelevant page: {current_url}")
                continue
            
            
            structured_data = self.extract_data(current_url, content, instructions)
            
            
            if structured_data and not self._is_duplicate_content(
                structured_data.get('extracted_content'), 
                self.extracted_data
            ):
                self.extracted_data.append(structured_data)
                
                if self.save_intermediate:
                    self.save_results(f"rufus_intermediate_{len(self.extracted_data)}.json")
            else:
                self.logger.info(f"Skipping duplicate content from {current_url}")
            
            
            if current_depth < self.max_depth:
                for link in links:
                    try:
                        absolute_link = urljoin(current_url, link)
                        # Only follow links within the same domain
                        if urlparse(absolute_link).netloc == domain and absolute_link not in self.visited_urls:
                            self.to_visit.append((absolute_link, current_depth + 1))
                    except Exception as e:
                        self.logger.warning(f"Error processing link {link}: {e}")
        
        
        self.extracted_data = self._post_process_extracted_data()
        
        self.logger.info(f"Crawling completed. Processed {len(self.visited_urls)} pages, extracted {len(self.extracted_data)} documents.")
        return self.extracted_data
    
    def save_results(self, output_file="rufus_output.json"):
        """Save the extracted data to a file"""
        try:
            with open(output_file, "w") as f:
                json.dump(self.extracted_data, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving results to {output_file}: {e}")
    
    def close(self):
        """Close the Selenium driver"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Rufus has been shut down.")
            except Exception as e:
                self.logger.error(f"Error shutting down Rufus: {e}")
        else:
            self.logger.warning("No WebDriver instance to close.")

def main():
    parser = argparse.ArgumentParser(description="Rufus - Intelligent Web Data Extraction for LLMs")
    parser.add_argument("--url", type=str, help="URL to start crawling from")
    parser.add_argument("--instructions", type=str, help="Instructions for data extraction")
    parser.add_argument("--max_pages", type=int, default=3, help="Maximum number of pages to crawl")
    parser.add_argument("--max_depth", type=int, default=1, help="Maximum depth of links to follow")
    parser.add_argument("--output", type=str, default="rufus_output.json", help="Output file name")
    parser.add_argument("--log_file", type=str, help="Path to log file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--similarity", type=float, default=0.7, help="Similarity threshold for duplicate detection (0.0-1.0)")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results after each page")
    parser.add_argument("--disable-llm-relevance", action="store_true", help="Disable LLM-based relevance checking")
    
    args = parser.parse_args()
    
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    
    url = args.url
    if not url:
        url = input("Enter the URL to start crawling from: ")
    
    instructions = args.instructions
    if not instructions:
        instructions = input("Enter your instructions for data extraction: ")
    
    # Initialize Rufus
    rufus = Rufus(
        max_pages=args.max_pages, 
        max_depth=args.max_depth,
        log_level=log_level,
        log_file=args.log_file,
        similarity_threshold=args.similarity,
        save_intermediate=args.save_intermediate,
        llm_relevance_check=not args.disable_llm_relevance
    )
    
    try:
        rufus.crawl(url, instructions)
        
        rufus.save_results(args.output)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")
        rufus.save_results(args.output)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        # Try to save any data that was collected
        if rufus.extracted_data:
            rufus.save_results(args.output)
    
    finally:
        rufus.close()

if __name__ == "__main__":
    main() 