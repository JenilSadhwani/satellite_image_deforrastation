from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

app = Flask(__name__)

class RestorationMethod:
    def __init__(self, name, cost_per_meter):
        self.name = name
        self.cost_per_meter = cost_per_meter
        self.total_cost = 0
        self.annual_cost = 0

def calculate_deforestation(image_before, image_after):
    # Convert images to grayscale
    gray_before = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(image_after, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between the images
    diff = cv2.absdiff(gray_before, gray_after)

    # Apply Gaussian blur
    blurred_before = cv2.GaussianBlur(gray_before, (5, 5), 0)
    blurred_after = cv2.GaussianBlur(gray_after, (5, 5), 0)
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Threshold the difference image to get binary image
    _, thresh = cv2.threshold(blurred_diff, 25, 255, cv2.THRESH_BINARY)

    # Calculate contour area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = 0
    for contour in contours:
        contour_area += cv2.contourArea(contour)

    # Calculate deforestation percentage
    total_pixels = gray_before.shape[0] * gray_before.shape[1]
    deforestation_percentage = (contour_area / total_pixels) * 100

    # Calculate mean pixel intensity
    mean_pixel_intensity = np.mean(gray_before)

    return deforestation_percentage, mean_pixel_intensity, contour_area, thresh, blurred_before, blurred_after

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_before = request.files['image_before']
    image_after = request.files['image_after']

    image_before_path = os.path.join('uploads', image_before.filename)
    image_after_path = os.path.join('uploads', image_after.filename)

    image_before.save(image_before_path)
    image_after.save(image_after_path)

    # Read images using cv2
    image_before_cv = cv2.imread(image_before_path)
    image_after_cv = cv2.imread(image_after_path)

    deforestation_percentage, mean_pixel_intensity, contour_area, thresh, blurred_before, blurred_after = calculate_deforestation(image_before_cv, image_after_cv)

    # Create feature matrix and target vector
    X = np.array([[deforestation_percentage, mean_pixel_intensity, contour_area]])
    y = np.array([deforestation_percentage])

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict deforestation percentage on the training data
    y_pred = model.predict(X)

    # Calculate the mean squared error
    mse = mean_squared_error(y, y_pred)

    # Plot the predicted vs actual deforestation percentage
    plt.scatter(y, y, color='blue', label='Actual', alpha=0.5)
    plt.scatter(y, y_pred, color='red', label='Predicted', alpha=0.5)
    plt.title('Actual vs Predicted Deforestation Percentage')
    plt.xlabel('Actual Deforestation Percentage')
    plt.ylabel('Predicted Deforestation Percentage')
    plt.legend()
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Restoration methods
    methods = [
        RestorationMethod('Agroforestry', 50),
        RestorationMethod('Reforestation with native species', 60),
        RestorationMethod('Soil conservation techniques', 40),
        RestorationMethod('Afforestation with local species', 55),
        RestorationMethod('Terracing', 45),
        RestorationMethod('Water conservation and management', 35),
        RestorationMethod('Bioengineering techniques', 70),
        RestorationMethod('Grassland restoration', 30),
        RestorationMethod('Wetland restoration', 65),
        RestorationMethod('Urban forestry', 75),
    ]

    total_cost = 0
    for method in methods:
        # Cost per meter per year
        method.annual_cost = method.cost_per_meter * contour_area
        method.total_cost = method.annual_cost * 1  # considering it takes at least a year to complete
        total_cost += method.total_cost

    return render_template('result.html',
                           deforestation_percentage=deforestation_percentage,
                           mean_pixel_intensity=mean_pixel_intensity,
                           contour_area=contour_area,
                           mse=mse,
                           y=y,
                           y_pred=y_pred,
                           titles=['Before Blurring', 'After Blurring', 'Thresholded Difference'],
                           image_paths=[os.path.join('static', 'temp_images', f'{title.replace(" ", "_")}.png') for title in ['Before Blurring', 'After Blurring', 'Thresholded Difference']],
                           methods=methods,
                           total_cost=total_cost
                          )

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists(os.path.join('static', 'temp_images')):
        os.makedirs(os.path.join('static', 'temp_images'))
    app.run(debug=True)





