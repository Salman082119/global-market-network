import time

from selenium import webdriver
from selenium.webdriver.edge.options import Options

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1680,2600")
opts.add_argument("--disable-gpu")

driver = webdriver.Edge(options=opts)
try:
    driver.get("http://localhost:8600/")
    time.sleep(20)
    driver.save_screenshot("shot_dashboard_latest.png")
    print("saved latest view")

    driver.get("http://localhost:8600/?asof_date=2020-03-20")
    time.sleep(15)
    driver.save_screenshot("shot_dashboard_covid.png")
    print("saved COVID view")
finally:
    driver.quit()
