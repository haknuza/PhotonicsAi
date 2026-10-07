import numpy as np
import pandas as pd

x = np.linspace(0, 10, 500)
y_noise = np.sin(x) + 0.1 * np.random.normal(size=x.size)

df = pd.DataFrame({"x": x, "sine_with_noise": y_noise})

# Save to CSV (index=False prevents saving row numbers)
df.to_csv("wave_data.csv", index=False)