import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1680,3200")
opts.add_argument("--disable-gpu")
driver = webdriver.Edge(options=opts)
try:
    driver.get("http://localhost:8601/")
    time.sleep(18)
    new_ones = {"move": 8, "news": 9, "week52": 10, "heatmap": 11, "deep": 12}
    for nm, idx in new_ones.items():
        try:
            tbs = driver.find_elements(By.CSS_SELECTOR,
                '[data-testid="stTabs"] button[data-baseweb="tab"]')
            print(f"tab count: {len(tbs)}")
            tbs[idx].click()
            time.sleep(8)
            driver.save_screenshot(f"advisor_shot_{nm}.png")
            print(nm, "shot ok")
        except Exception as e:
            print(nm, "ERR", str(e)[:120])
finally:
    driver.quit()
