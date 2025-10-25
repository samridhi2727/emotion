import numpy as np
import cv2
from PIL import Image

def detect_faces_and_emotions(
    img_input, face_cascade, model, labels, threshold=0.5, selected_emotions=None
):
    """
    Detect faces and predict emotions using a pre-trained model.
    
    Parameters:
    - img_input: PIL.Image, file buffer, or numpy.ndarray
    - face_cascade: OpenCV Haar Cascade for face detection
    - model: Trained CNN model for emotion classification
    - labels: List of emotion labels
    - threshold: Minimum probability required to include an emotion
    - selected_emotions: Optional list of emotions to filter
    
    Returns:
    - List of tuples [(emotion_label, probability_percent), ...], no duplicates,
      sorted internally by descending probability per face detected.
    """
    
    # Convert input to numpy array (BGR for OpenCV)
    if isinstance(img_input, Image.Image):
        img = np.array(img_input)
    elif hasattr(img_input, 'read'):  # Streamlit file buffer
        img = np.array(Image.open(img_input))
    elif isinstance(img_input, np.ndarray):
        img = img_input
    else:
        raise ValueError("Unsupported image format. Provide PIL, np.ndarray, or file buffer.")
    
    # Convert RGB to BGR if needed
    if img.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError("Only 3-channel or 4-channel images are supported.")
    
    # Detect faces
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    
    combined_results = {}  # dict to keep max probability per emotion across all faces
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        
        sharpened = cv2.filter2D(roi_resized, -1, sharpen_kernel)
        roi_input = sharpened.reshape(1, 48, 48, 1).astype('float32') / 255.0
        
        preds = model.predict(roi_input, verbose=0)[0]  # array of probabilities
        
        # Filter predicted emotions per selected_emotions if given
        # Build a dict of emotion -> probability for this face (above threshold)
        face_emotions = {}
        for i, prob in enumerate(preds):
            emotion = labels[i]
            if selected_emotions and (emotion not in selected_emotions):
                continue
            if prob >= threshold:
                face_emotions[emotion] = prob
        
        # If no emotion above threshold, fallback to top predicted emotion regardless
        if not face_emotions:
            top_i = int(np.argmax(preds))
            top_emotion = labels[top_i]
            if (not selected_emotions) or (top_emotion in selected_emotions):
                face_emotions[top_emotion] = preds[top_i]
        
        # Update combined_results with max probability per emotion (across faces)
        for emotion, prob in face_emotions.items():
            if emotion in combined_results:
                combined_results[emotion] = max(combined_results[emotion], prob)
            else:
                combined_results[emotion] = prob
    
    # Convert to list and sort descending by probability
    sorted_results = sorted(
        [(e, p * 100) for e, p in combined_results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results
