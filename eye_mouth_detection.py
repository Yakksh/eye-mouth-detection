import cv2
import dlib
import screeninfo

# Load pre-trained facial landmark detector from dlib
predictor_path = 'shape_predictor_68_face_landmarks.dat'  # Path to pre-trained model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to calculate bounding box coordinates
def calculate_bounding_box(points):
    x = min(points, key=lambda p: p[0])[0]
    y = min(points, key=lambda p: p[1])[1]
    w = max(points, key=lambda p: p[0])[0] - x
    h = max(points, key=lambda p: p[1])[1] - y
    return (x, y, w, h)

# Function to detect facial landmarks and calculate bounding boxes
def detect_facial_landmarks(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Get screen resolution
    screen_info = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen_info.width, screen_info.height

    # Calculate resizing factor to fit image on screen
    resize_factor = min(screen_width / image.shape[1], screen_height / image.shape[0])

    # Resize image
    resized_image = cv2.resize(image, (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor)))

    # Convert resized image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract coordinates for eyes and mouth
        left_eye = [(landmarks.part(36).x, landmarks.part(36).y),
                    (landmarks.part(39).x, landmarks.part(39).y)]
        right_eye = [(landmarks.part(42).x, landmarks.part(42).y),
                     (landmarks.part(45).x, landmarks.part(45).y)]
        mouth = [(landmarks.part(48).x, landmarks.part(48).y),
                 (landmarks.part(54).x, landmarks.part(54).y)]

        # Calculate bounding boxes
        left_eye_bbox = calculate_bounding_box(left_eye)
        right_eye_bbox = calculate_bounding_box(right_eye)
        mouth_bbox = calculate_bounding_box(mouth)

        # Scale bounding boxes to original image size
        left_eye_bbox = tuple(int(coord / resize_factor) for coord in left_eye_bbox)
        right_eye_bbox = tuple(int(coord / resize_factor) for coord in right_eye_bbox)
        mouth_bbox = tuple(int(coord / resize_factor) for coord in mouth_bbox)
        print(f"Left Eye bbox coordinates: {left_eye_bbox}")
        print(f"Right Eye bbox coordinates: {right_eye_bbox}")
        print(f"Mouth bbox coordinates: {mouth_bbox}")
        
        # Draw bounding boxes on the image
        cv2.rectangle(image, (left_eye_bbox[0], left_eye_bbox[1]),
                      (left_eye_bbox[0] + left_eye_bbox[2], left_eye_bbox[1] + left_eye_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (right_eye_bbox[0], right_eye_bbox[1]),
                      (right_eye_bbox[0] + right_eye_bbox[2], right_eye_bbox[1] + right_eye_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (mouth_bbox[0], mouth_bbox[1]),
                      (mouth_bbox[0] + mouth_bbox[2], mouth_bbox[1] + mouth_bbox[3]), (0, 255, 0), 2)

    # Display the resized image with bounding boxes
    cv2.imshow('Facial Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in range(0,9):
    print(f'Coordinates for 0000{i}.png')
    image_path = f'samples/0000{i}.png'
    detect_facial_landmarks(image_path)