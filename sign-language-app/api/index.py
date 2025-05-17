from http.server import BaseHTTPRequestHandler
import sys
import os

# Add the parent directory to the sys.path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write('Sign Language Recognition API'.encode())
        return 