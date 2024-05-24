import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Adjust SIFT parameters here
sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=5)

def convert_to_polar(image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    max_radius = np.hypot(center[0], center[1])
    polar_image = cv2.linearPolar(image, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return polar_image, center, max_radius

def remove_background(image):
    # Create a mask initialized with zeros (background)
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Define background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define a rectangle enclosing the foreground object
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    
    # Run GrabCut algorithm to segment foreground from background
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify the mask to mark probable foreground and definite background pixels
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the original image
    result = image * mask2[:, :, np.newaxis]
    
    return result



def detect_rotational_symmetry(image):
    # Convert to polar coordinates
    polar_image, center, max_radius = convert_to_polar(image)
    
    # Detect SIFT features
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(polar_image, None)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Prepare data for polar domain plot
    houghr = []
    houghth = []
    for match in matches:
        point = kp1[match.queryIdx].pt
        polar_point = kp2[match.trainIdx].pt
        
        theta = np.arctan2(point[1] - center[1], point[0] - center[0])
        r = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        
        houghr.append(r)
        houghth.append(np.degrees(theta))
    
    # Create an intensity image that highlights SIFT features
    intensity_image = np.zeros_like(image)
    for kp in kp1:
        x, y = np.int32(kp.pt)
        cv2.circle(intensity_image, (x, y), 5, (255, 255, 255), -1)
    
    # Plotting
    plt.figure(figsize=(15, 7))
    
    # Original image with SIFT features
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(intensity_image, cv2.COLOR_BGR2RGB))
    plt.title("Intensity Image with SIFT Features")
    plt.axis('off')
    
    # Symmetry detected
    plt.subplot(232)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for match in matches:
        point = kp1[match.queryIdx].pt
        plt.scatter(point[0], point[1], c='r', s=5)
    plt.title("Symmetry Detected")
    plt.axis('off')
    
    # Polar domain plot
    plt.subplot(233, projection='polar')
    plt.scatter(np.radians(houghth), houghr, s=5)
    plt.title("Polar Domain")
    
    
    
    plt.tight_layout()
    plt.show()

def main():
    argc = len(sys.argv)
    if argc != 2:
        print("Usage: python3 detect.py IMAGE")
        return
    
    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error: Could not read image '{sys.argv[1]}'")
        return
    
    # Remove background
    image_without_bg = remove_background(image)

    # Preprocess the image to increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayscale_image = cv2.cvtColor(image_without_bg, cv2.COLOR_BGR2GRAY)
    contrast_enhanced_image = clahe.apply(grayscale_image)

    # Resize the image
    resized_image = cv2.resize(contrast_enhanced_image, None, fx=2.0, fy=2.0)

    # Detect rotational symmetry
    detect_rotational_symmetry(resized_image)

if __name__ == "__main__":
    main()
