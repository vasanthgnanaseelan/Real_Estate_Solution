import logging
import pandas as pd
import matplotlib.pyplot as plt

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Pandas Options
pd.set_option('display.max_columns', None)

# Matplotlib Style
plt.style.use('ggplot')
