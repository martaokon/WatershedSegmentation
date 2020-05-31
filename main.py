from data import *
from watershed import *

# Load data
imgs = loadImgs()
ws = watershedSegmentation(imgs)
# Apply watershed segmentation
imgs_final = ws.segmentize()
# Save results
saveImgs(imgs_final)
# Show results
showImgs(imgs_final)
