import os
os.environ["KAGGLEHUB_CACHE"] = "D:/kaggle_cache"

import kagglehub

path = kagglehub.dataset_download(
"yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018"
)

print("Dataset downloaded successfully!")
print("Dataset location:", path)
