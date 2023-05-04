import cv2
import numpy as np

def selective_search(image, mode='fast'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    elif mode == 'quality':
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError("Invalid mode. Choose either 'fast' or 'quality'")

    rects = ss.process()
    return rects

def filter_rects(rects, min_area=500, max_proposals=100):
    filtered_rects = []

    for rect in rects:
        x, y, w, h = rect
        area = w * h
        if area >= min_area:
            filtered_rects.append(rect)

    filtered_rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return filtered_rects[:max_proposals]

if __name__ == "__main__":
    image_path = "dog.jpeg"
    image = cv2.imread(image_path)

    mode = 'fast'
    rects = selective_search(image, mode)

    # Filter the region proposals
    min_area = 500
    max_proposals = 10
    filtered_rects = filter_rects(rects, min_area, max_proposals)

    output = image.copy()
    for i, rect in enumerate(filtered_rects):
        x, y, w, h = rect
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Selective Search Results", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
