from selenium.webdriver.chrome.options import Options


def pytest_setup_options():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    return options
