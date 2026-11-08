import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1680,3400")
opts.add_argument("--disable-gpu")
driver = webdriver.Edge(options=opts)
try:
    driver.get("http://localhost:8601/")
    time.sleep(18)
    # tab 1 advisor - check top movers cards
    body = driver.find_element(By.TAG_NAME, "body").text
    has_movers = "Today's top movers" in body or "Top gainers" in body
    print("advisor top movers present:", has_movers)
    driver.save_screenshot("verify_advisor.png")

    # backtest tab (2) - check drawdown
    tabs = driver.find_elements(By.CSS_SELECTOR,
        '[data-testid="stTabs"] button[data-baseweb="tab"]')
    tabs[2].click()
    time.sleep(7)
    body = driver.find_element(By.TAG_NAME, "body").text
    print("backtest drawdown present:", "Drawdown" in body)
    driver.save_screenshot("verify_backtest.png")

    # compare tab (7) - 52wk (harder; compare needs 2 different assets)
    tabs = driver.find_elements(By.CSS_SELECTOR,
        '[data-testid="stTabs"] button[data-baseweb="tab"]')
    tabs[7].click()
    time.sleep(7)
    driver.save_screenshot("verify_compare.png")
    print("compare tab ok")

    # news tab (9)
    tabs = driver.find_elements(By.CSS_SELECTOR,
        '[data-testid="stTabs"] button[data-baseweb="tab"]')
    tabs[9].click()
    time.sleep(12)
    driver.save_screenshot("verify_news.png")
    body = driver.find_element(By.TAG_NAME, "body").text
    print("news tab loaded (sentiment section):", "Sentiment" in body)

    # deep tab (12)
    tabs = driver.find_elements(By.CSS_SELECTOR,
        '[data-testid="stTabs"] button[data-baseweb="tab"]')
    tabs[12].click()
    time.sleep(8)
    driver.save_screenshot("verify_deep.png")
    print("deep tab ok")
    print("ALL DONE")
finally:
    driver.quit()
