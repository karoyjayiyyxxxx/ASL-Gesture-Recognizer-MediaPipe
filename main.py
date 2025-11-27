#--coding:utf-8 --

import json
import sys
import os
import cv2
import time
import threading
import pygame
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from autocorrect import Speller
from typing import List, Tuple
from google import genai

# Kivy imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import ObjectProperty

# Local imports
from utils import load_model, save_gif, save_video
from utils import calc_landmark_list, draw_landmarks, draw_info_text

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Constants
class Colors:
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)

class Config:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    MAX_HANDS = 1
    MIN_DETECTION_CONFIDENCE = 0.6
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_PATH = "./classifier"
    FONT_PATH = "NotoSansTC-Regular.ttf"
    FONT_SIZE = 40#20
    TIMING_THRESHOLD = 8
    AUTOCORRECT = True
    SAVE_GIF = False
    SAVE_VIDEO = False
    FPS = 30

    @classmethod
    def get_model_paths(cls) -> Tuple[str, str]:
        """Returns the paths for letter and number models"""
        return (
            f"{cls.MODEL_PATH}/classify_letter_model.p", 
            f"{cls.MODEL_PATH}/classify_number_model.p"
        )

# Define the UI layout first
class GestureAppLayout(BoxLayout):
    """Main layout for the Gesture Recognition App."""
    
    # Use Kivy properties for better widget access
    camera_image = ObjectProperty(None)
    status_label = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(GestureAppLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Camera display
        self.camera_box = BoxLayout(orientation='vertical', size_hint=(1, 0.7))
        self.camera_image = KivyImage(allow_stretch=True, keep_ratio=True)
        self.camera_box.add_widget(self.camera_image)
        self.add_widget(self.camera_box)
        
        # Status label
        self.status_label = Label(text='Ready', font_name = Config.FONT_PATH, size_hint=(1, 0.1))
        self.add_widget(self.status_label)
        
        # Buttons row 1
        button_row1 = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.mode_button = Button(text='切換字母/數字', font_name = Config.FONT_PATH)#'Toggle A/1')
        self.clear_button = Button(text='清除', font_name = Config.FONT_PATH)#'Clear')
        self.delete_button = Button(text='清空', font_name = Config.FONT_PATH)#'Delete Last')
        
        button_row1.add_widget(self.mode_button)
        button_row1.add_widget(self.clear_button)
        button_row1.add_widget(self.delete_button)
        self.add_widget(button_row1)
        
        # Buttons row 2
        button_row2 = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        
        self.translate_button = Button(text='翻譯/發音', font_name = Config.FONT_PATH)
        self.save_button = Button(text='儲存', font_name = Config.FONT_PATH)
        
        button_row2.add_widget(self.translate_button)
        button_row2.add_widget(self.save_button)
        self.add_widget(button_row2)
    
    def update_status(self, text):
        """Update the status label text."""
        self.status_label.text = text


class GestureRecognizer:
    """Main class for gesture recognition and translation."""
    
    def __init__(self):
        """Initialize the gesture recognizer."""
        self.timing_threshold = Config.TIMING_THRESHOLD
        self.autocorrect = Config.AUTOCORRECT
        self.save_gif = Config.SAVE_GIF
        self.save_video = Config.SAVE_VIDEO
        self.fps = Config.FPS
        
        # Automatically determine webcam dimensions based on device
        self.webcam_width = 640  # Default mobile width 
        self.webcam_height = 480  # Default mobile height
        
        # Output tracking
        self._output = [[], []]
        self.output = []
        self.current_hand = 0
        self.number_mode = False
        self.translated_results = []
        self.final_result = ""
        
        # Initialize spell checker
        self.spell = Speller(lang='en')
        
        # Store kivy widgets - will be set by the app
        self.camera_image = None
        self.status_label = None
        
        # Load font for text rendering
        try:
            self.font = ImageFont.truetype(Config.FONT_PATH, Config.FONT_SIZE, encoding="utf-8")
        except IOError:
            print(f"Warning: Could not load font {Config.FONT_PATH}. Using default font.")
            self.font = ImageFont.load_default()
        
        # API key (from environment variable for security)
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not self.google_api_key:
            print("Warning: GOOGLE_API_KEY environment variable not set.")
        
        # Load models
        model_letter_path, model_number_path = Config.get_model_paths()
        self.letter_model = load_model(model_letter_path)
        self.number_model = load_model(model_number_path)
        
        # Initialize TTS engine
        try:
            pygame.mixer.init()
            self.tts_engine = "pygame"  # Just a marker to know we're using pygame
        except Exception as e:
            print(f"Warning: Could not initialize audio system: {e}")
            self.tts_engine = None
        
        # Initialize video capture and hands
        self.capture = None
        self.hands = mp_hands.Hands(
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
            max_num_hands=Config.MAX_HANDS
        )
        self.frame_array = []
        self.is_running = False

    def setup_capture(self):
        """Setup the video capture from webcam."""
        self.capture = cv2.VideoCapture(0)  # Use default camera
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.capture.isOpened():
            raise ValueError("Could not open webcam")
            
        return self.capture

    def release_resources(self):
        """Release camera and other resources."""
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        self.hands.close()
        if self.tts_engine is not None:
            try:
                self.tts_engine.stop()
            except Exception:
                pass

    def get_output(self, idx: int) -> None:
        """Process the output buffer and append complete gestures to output list."""
        key = []
        for i in range(len(self._output[idx])):
            character = self._output[idx][i]
            counts = self._output[idx].count(character)

            # Add character to key if it exceeds timing threshold
            if (not key or character != key[-1]) and counts > self.timing_threshold:
                key.append(character)

        # Convert key characters to text
        text = ""
        for character in key:
            if character == "?":
                continue
            text += str(character).lower()

        # Apply autocorrect if enabled
        if text and self.autocorrect:
            text = self.spell(text)

        # Add word to output list
        if text:
            self._output[idx] = []
            self.output.append(text.title())
            self.update_status(f"Added: {text.title()}")

    def recognize_gesture(self, image: np.ndarray, results) -> np.ndarray:
        """Recognize hand gestures in the image and process the results."""
        is_increased = False
        is_decreased = False
        
        # Check hand count changes
        if self.current_hand != 0:
            if results.multi_hand_landmarks is None:
                is_decreased = True
            else:
                new_hand_count = len(results.multi_hand_landmarks)
                if new_hand_count > self.current_hand:
                    is_increased = True
                elif new_hand_count < self.current_hand:
                    is_decreased = True

        _gesture = []
        
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            
            for idx in reversed(range(len(results.multi_hand_landmarks))):
                current_select_hand = results.multi_hand_landmarks[idx]
                handness = results.multi_handedness[idx].classification[0].label
                
                # Process landmarks
                landmark_list = calc_landmark_list(image, current_select_hand)
                image = draw_landmarks(image, landmark_list)
                
                # Get coordinates and process hand data
                data_aux = []
                x_values = [lm.x for lm in current_select_hand.landmark]
                y_values = [lm.y for lm in current_select_hand.landmark]
                
                # Get bounding box coordinates
                min_x = int(min(x_values) * w)
                max_x = int(max(x_values) * w)
                min_y = int(min(y_values) * h)
                max_y = int(max(y_values) * h)
                
                # Draw text information
                cv2.putText(image, f"Hand No. #{idx}", (min_x - 10, max_y + 30), 
                            Config.FONT, 1, Colors.GREEN, 2)
                cv2.putText(image, f"{handness} Hand", (min_x - 10, max_y + 60), 
                            Config.FONT, 1, Colors.GREEN, 2)
                
                # Flip left hand to right hand
                if handness == 'Left':
                    x_values = list(map(lambda x: 1 - x, x_values))
                    min_x -= 10
                
                # Create data points for model prediction
                for i in range(len(current_select_hand.landmark)):
                    data_aux.append(x_values[i] - min(x_values))
                    data_aux.append(y_values[i] - min(y_values))
                
                # Predict gesture based on mode
                if not self.number_mode:
                    prediction = self.letter_model.predict([np.asarray(data_aux)])
                    gesture = str(prediction[0]).title()
                    gesture = gesture if gesture != 'Unknown_Letter' else '?'
                else:
                    prediction = self.number_model.predict([np.asarray(data_aux)])
                    gesture = str(prediction[0]).title()
                    gesture = gesture if gesture != 'Unknown_Number' else '?'
                
                # Draw bounding box and gesture info
                cv2.rectangle(image, (min_x - 20, min_y - 10), 
                              (max_x + 20, max_y + 10), Colors.BLACK, 4)
                image = draw_info_text(image, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], gesture)
                
                _gesture.append(gesture)
        
        # Process hand count changes
        if is_decreased and self.current_hand == 1:
            self.get_output(0)
        elif results.multi_hand_landmarks is not None:
            if _gesture:  # Only append if there's a gesture detected
                self._output[0].append(_gesture[0])
        
        # Update current hand count
        self.current_hand = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            
        return image

    def get_gemini_response(self, text: str) -> List[str]:
        """Get translation suggestions from Gemini API."""
        self.update_status("Translating...")
        if not self.google_api_key:
            self.update_status("No API key available")
            return ["Warning: No API key provided for translation"]
            
        prompt = (
            f"if the sentence ['{text.lower()}'] is not complete, please complete the "
            f"top 5 possible sentences in Traditional Chinese? with this schema: "
            f"{{ \"Sentence\": str}} but without any explanation"
        )
        
        try:
            client = genai.Client(api_key=self.google_api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            # Process response
            text = response.text.replace("```json", "").replace("```", "")
            list_translated_result = json.loads(text)
            
            self.update_status("Translation complete")

            list_translated_result = [f"{i + 1}.{list_translated_result[i]['Sentence']}" for i in range(len(list_translated_result))]

            print(list_translated_result)
            return list_translated_result
                   
        except Exception as e:
            error_msg = str(e)
            self.update_status(f"Translation error: {error_msg[:20]}...")
            print(f"Error in Gemini API call: {e}")
            return [f"Translation error: {error_msg}"]

    
    def speak(self, selected_text):

        try:
            from gtts import gTTS
            import tempfile
        
            self.update_status("Generating speech...")
        
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
            
            # Generate speech
            tts = gTTS(text=selected_text, lang='zh-TW')  # Use Traditional Chinese
            tts.save(temp_filename)
        
            # Play the speech
            self.update_status("Speaking...")
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
        
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            self.update_status("Done speaking")
        except Exception as e:
            print(f"TTS error: {str(e)}")
            self.update_status(f"Speech error: {str(e)[:20]}...")


    def overlay_text_on_image(self, translated_results: List[str], image: np.ndarray) -> np.ndarray:
        """Add translated text to the image."""
        if not translated_results:
            return image
            
        # Convert OpenCV image to PIL image for text rendering
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Draw text with proper positioning
        # Use a semi-transparent background for better readability
        h, w, _ = image.shape
        text = "\n".join(translated_results)
        
        # Calculate text size
        font_size = self.font.size
        line_height = font_size + 4
        num_lines = len(translated_results)
        text_height = line_height * num_lines
        
        # Create background rectangle
        x1, y1 = 20, 20
        x2, y2 = w - 20, y1 + text_height + 20
        
        # Draw semi-transparent background
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, 160))
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # Draw text
        draw.text((x1 + 10, y1 + 10), text, Colors.WHITE, font=self.font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    def update_status(self, text):
        """Update the status label text safely."""
        if self.status_label:
            # Use Clock to ensure UI updates happen on the main thread
            Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', text), 0)

    # Button functions
    def toggle_mode(self, instance=None):
        """Toggle between letter and number mode."""
        self.number_mode = not self.number_mode
        self.update_status(f"Mode: {'Number' if self.number_mode else 'Letter'}")

    def clear_output(self, instance=None):
        """Clear the current output text."""
        self.output.clear()
        self.update_status("Output cleared")

    def delete_last_word(self, instance=None):
        """Delete the last word from the output."""
        if self.output:
            word = self.output.pop()
            self.update_status(f"Deleted: {word}")
        else:
            self.update_status("Nothing to delete")

    def translate_or_speak(self, instance=None):
        """Translate text or speak a selected translation."""
        if not self.number_mode:
            # Switch to number mode and get translations
            self.final_result = ' '.join(self.output)
            if not self.final_result:
                self.update_status("No text to translate")
                return
                
            self.output.clear()
            self.number_mode = True
            self.update_status(f"Translating: {self.final_result}")
            self.translated_results = self.get_gemini_response(self.final_result)
        else:
            # Process number selection
            final_result = ''.join(self.output)
            if final_result.isnumeric():
                selected_number = int(final_result) - 1  # Adjust for 0-indexing
                if 0 <= selected_number < len(self.translated_results):
                    selected_text = self.translated_results[selected_number]
                    self.update_status(f"Speaking: {selected_text[:20]}...")
                    self.translated_results = []
                    if self.tts_engine is not None:
                        self.output.clear()
                        self.number_mode = False
                        threading.Thread(
                            target=self.speak, 
                            args=(selected_text,),
                            daemon=True
                        ).start()
                else:
                    self.update_status(f"Invalid selection: {selected_number + 1}")
            else:
                self.update_status("Please enter a valid number")
            self.output.clear()

    def save_results(self, instance=None):
        """Save current session as GIF and/or video."""
        if not self.frame_array:
            self.update_status("No frames to save")
            return
            
        self.update_status("Saving results...")
        
        try:
            save_gif(self.frame_array, "gesture_recognition.gif", fps=10)
            self.update_status("GIF saved")
            
            save_video(self.frame_array, "gesture_recognition.mp4", fps=self.fps)
            self.update_status("Video saved")
        except Exception as e:
            self.update_status(f"Save error: {str(e)[:20]}...")
            print(f"Error saving results: {e}")

    def process_frame(self, dt):
        """Process a single frame from the camera."""
        if not self.is_running or self.capture is None:
            return
            
        success, frame = self.capture.read()
        if not success:
            self.update_status("Camera error")
            return
            
        # Process image for hand detection
        frame = cv2.flip(frame, 1)  # Mirror image for selfie view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Process the frame
        output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        try:
            # Recognize gestures
            output_frame = self.recognize_gesture(output_frame, results)
            
            # Add translated text if available
            if self.number_mode and self.translated_results:
                output_frame = self.overlay_text_on_image(self.translated_results, output_frame)
            
            # Display output text
            output_text = str(self.output)
            output_size = cv2.getTextSize(output_text, Config.FONT, 0.5, 2)[0]
            cv2.rectangle(output_frame, (5, 0), (50 + output_size[0], 50 + output_size[1]), 
                         Colors.YELLOW, -1)
            cv2.putText(output_frame, output_text, (10, 15), Config.FONT, 0.5, Colors.BLACK, 2)
            
            # Display mode
            mode_text = f"Mode: {'Number' if self.number_mode else 'Letter'}"
            mode_size = cv2.getTextSize(mode_text, Config.FONT, 0.5, 2)[0]
            cv2.rectangle(output_frame, (5, 45), (10 + mode_size[0], 10 + mode_size[1]), 
                         Colors.YELLOW, -1)
            cv2.putText(output_frame, mode_text, (10, 40), Config.FONT, 0.5, Colors.BLACK, 2)
            
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"Error: {error}, line {exc_tb.tb_lineno}")
            self.update_status(f"Error: {str(error)[:20]}...")
        
        # Save frame for potential GIF/video
        self.frame_array.append(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
        
        # Convert the frame to texture for Kivy
        buf = cv2.flip(output_frame, 0)
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        
        # Update the image widget - use directly referenced widget
        if self.camera_image:
            self.camera_image.texture = texture


class GestureApp(App):
    """Kivy application for gesture recognition."""
    
    def build(self):
        """Build the application UI."""
        # Create the layout first
        layout = GestureAppLayout()
        
        # Create the recognizer
        self.recognizer = GestureRecognizer()
        
        # Set up camera based on window size
        window_width, window_height = Window.size
        aspect_ratio = 4/3  # Standard mobile camera aspect ratio
        
        # Calculate camera dimensions that fit the screen
        if window_width / window_height > aspect_ratio:
            # Screen is wider than camera aspect ratio
            self.recognizer.webcam_height = int(window_height * 0.7)  # Use 70% of height
            self.recognizer.webcam_width = int(self.recognizer.webcam_height * aspect_ratio)
        else:
            # Screen is taller than camera aspect ratio
            self.recognizer.webcam_width = int(window_width * 0.9)  # Use 90% of width
            self.recognizer.webcam_height = int(self.recognizer.webcam_width / aspect_ratio)
        
        # Connect button events directly to the recognizer methods
        layout.mode_button.bind(on_press=self.recognizer.toggle_mode)
        layout.clear_button.bind(on_press=self.recognizer.clear_output)
        layout.delete_button.bind(on_press=self.recognizer.delete_last_word)
        layout.translate_button.bind(on_press=self.recognizer.translate_or_speak)
        layout.save_button.bind(on_press=self.recognizer.save_results)
        
        # Give the recognizer direct references to the UI elements it needs
        self.recognizer.camera_image = layout.camera_image
        self.recognizer.status_label = layout.status_label
        
        return layout
        
    def on_start(self):
        """Start camera and processing when app starts."""
        try:
            self.recognizer.setup_capture()
            self.recognizer.is_running = True
            self.recognizer.update_status("Camera started")
            
            # Schedule the frame processing
            Clock.schedule_interval(self.recognizer.process_frame, 1.0/30.0)
        except Exception as e:
            print(f"Failed to start camera: {e}")
            if hasattr(self.root, 'status_label'):
                self.root.status_label.text = f"Camera error: {str(e)}"
            
    def on_stop(self):
        """Clean up resources when app stops."""
        self.recognizer.is_running = False
        Clock.unschedule(self.recognizer.process_frame)
        self.recognizer.release_resources()

if __name__ == '__main__':
    try:
        # Set Kivy environment variables for better mobile experience
        os.environ['KIVY_WINDOW'] = 'sdl2'
        os.environ['KIVY_GL_BACKEND'] = 'sdl2'
        os.environ['KIVY_IMAGE'] = 'sdl2,pil'
        
        # Run the app
        GestureApp().run()
    except Exception as e:
        print(f"Application error: {e}")