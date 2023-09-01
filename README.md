# YOLOv7 Object Detection with Flask

![Demo](https://github.com/abdul-rehman18/T-Shirt-Color-Live-Prediction-Using-Webcam-on-Yolov7/blob/master/github_tshirt.mp4)

This project demonstrates real-time object detection using the YOLOv7 model with a Flask web application. You can upload images or capture images from your webcam to perform object detection.

## Features

- **Real-Time Object Detection**: Detect objects in images captured from your webcam in real-time.
- **Upload Images**: Upload images from your device and see the detected objects.
- **Custom YOLOv7 Model**: This project uses a custom-trained YOLOv7 model for object detection.
- **User-Friendly Interface**: The web interface is designed to be user-friendly and easy to use.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Open your web browser and go to `http://localhost:5000` to access the application.

## Usage

1. **Webcam Capture**: Click the "Capture Image" button to capture an image from your webcam and perform object detection.

2. **Upload Images**: Click the "Upload Image" button to upload an image from your device and see the detected objects.

3. **Real-Time Webcam Feed**: Access the real-time webcam feed by visiting `/webcam`.

## Results

After performing object detection, the detected objects will be highlighted in the image, and the results will be displayed on the screen. You can also test another image by clicking the "Test Another Image" button.

## Model Details

This project uses a custom-trained YOLOv7 model for object detection. The model is loaded using [PyTorch Hub](https://pytorch.org/hub/wongkinyiu_yolov7_custom/), and it's based on the `best.pt` weight file.

## Credits

- YOLOv7 Model: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- Flask: [Flask Documentation](https://flask.palletsprojects.com/)

Feel free to contribute, open issues, or provide feedback to improve this project further.

Happy detecting!

---
