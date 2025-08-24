import os
import datetime
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import time

# The URL of the AGMARKNET report page you want to scrape
URL = "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx"

# Define the search criteria.
# These values correspond to the options in the dropdown menus.
# You can change these to scrape different data.
SEARCH_STATE = "Tamil Nadu"
SEARCH_COMMODITY = "Tomato"

def scrape_data(url):
    """
    Scrapes a specified URL using Selenium to handle dynamic content,
    simulates user input to filter data, and extracts the resulting market price data.
    """
    print("Starting Selenium WebDriver...")
    # Setup Chrome options to run headless (without a visible browser window)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize the WebDriver
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        # Wait for the state dropdown to be clickable and loaded
        print("Waiting for page to load...")
        wait = WebDriverWait(driver, 20)
        state_dropdown = wait.until(EC.element_to_be_clickable((By.NAME, 'ctl00$Content$ddlState')))
        
        # Select the desired state
        print(f"Attempting to select state: {SEARCH_STATE}")
        state_select = Select(state_dropdown)
        try:
            state_select.select_by_visible_text(SEARCH_STATE)
        except Exception:
            print(f"Warning: State '{SEARCH_STATE}' not found. Selecting the first available state.")
            # Select the first option that isn't the '--Select--' placeholder
            state_select.select_by_index(1) 
            SEARCH_STATE_ACTUAL = state_select.first_selected_option.text
            print(f"Selected State: {SEARCH_STATE_ACTUAL}")

        # Wait for the commodity dropdown to be populated and clickable
        print("Waiting for commodity list to populate...")
        commodity_dropdown = wait.until(EC.element_to_be_clickable((By.NAME, 'ctl00$Content$ddlCommodity')))
        
        # Select the desired commodity
        print(f"Attempting to select commodity: {SEARCH_COMMODITY}")
        commodity_select = Select(commodity_dropdown)
        try:
            commodity_select.select_by_visible_text(SEARCH_COMMODITY)
        except Exception:
            print(f"Warning: Commodity '{SEARCH_COMMODITY}' not found. Selecting the first available commodity.")
            # Select the first option that isn't the '--Select--' placeholder
            commodity_select.select_by_index(1)
            SEARCH_COMMODITY_ACTUAL = commodity_select.first_selected_option.text
            print(f"Selected Commodity: {SEARCH_COMMODITY_ACTUAL}")
        
        # Click the "Go" button to submit the form
        print("Submitting the form...")
        go_button = driver.find_element(By.NAME, 'ctl00$Content$btnSubmit')
        go_button.click()
        
        # Wait for the results table to appear
        print("Waiting for results table to appear...")
        wait.until(EC.presence_of_element_located((By.ID, 'ctl00_Content_GridView1')))
        
        # Parse the HTML content of the results page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find the data table
        table = soup.find('table', {'id': 'ctl00_Content_GridView1'})
        
        if not table:
            print("Could not find the data table on the page.")
            return []

        # Find all table rows (tr) and skip the header row
        rows = table.find_all('tr')[1:]

        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 7:
                try:
                    record = {
                        'state': cols[0].text.strip(),
                        'district': cols[1].text.strip(),
                        'market': cols[2].text.strip(),
                        'commodity': cols[3].text.strip(),
                        'variety': cols[4].text.strip(),
                        'modal_price': float(cols[7].text.strip().replace(',', ''))
                    }
                    data.append(record)
                except (ValueError, IndexError) as e:
                    print(f"Skipping row due to parsing error: {e}")
                    continue

        return data
        
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return []
    finally:
        if driver:
            print("Closing WebDriver...")
            driver.quit()

def save_data_to_json(data):
    """
    Saves the scraped data to a JSON file with a timestamped name.
    """
    if not data:
        print("No data to save.")
        return

    today = datetime.date.today().strftime('%Y-%m-%d')
    filename = f'market_prices_{today}.json'
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Successfully saved {len(data)} records to {filename}")

if __name__ == "__main__":
    print("Starting daily market price scrape...")
    market_data = scrape_data(URL)
    save_data_to_json(market_data)
    print("Scrape complete.")
