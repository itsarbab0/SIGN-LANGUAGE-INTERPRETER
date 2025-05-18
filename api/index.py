from http.server import BaseHTTPRequestHandler
import sys, os
from app import app

def handler(request):
    """Serverless function handler for Vercel"""
    return app(request)
