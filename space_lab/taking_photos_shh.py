import io
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from picamera import PiCamera

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = threading.Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        try:
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
        except Exception:
            pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "high_res_photos")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

camera = PiCamera(resolution=(4056, 3040), framerate=12)
output = StreamingOutput()

# Start the background stream on Port 2 (Resized for smooth network performance)
camera.start_recording(output, format='mjpeg', splitter_port=2, resize=(640, 480))

server = ThreadedHTTPServer(('', 8000), StreamingHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

print("\n" + "="*40)
print(f"LIVE VIEW: http://<your_pi_ip>:8000")
print("="*40 + "\n")

try:
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    count = len(existing_files)
    while True:
        cmd = input("Press ENTER to capture 12MP Photo ('q' to quit): ")
        
        if cmd.lower() == 'q':
            break
            
        filename = f"highres_{count}.jpg"
        save_path = os.path.join(output_folder, filename)
        
        # Capture the massive high-res file
        # The stream will pause for a second while the ISP processes the 12MP data
        camera.capture(save_path, quality=100)
        
        print(f"Captured: {filename}")
        count += 1

finally:
    camera.stop_recording(splitter_port=2)
    camera.close()
    server.shutdown()
