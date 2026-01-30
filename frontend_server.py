#!/usr/bin/env python3

import http.server
import socketserver
import os
import threading
import webbrowser

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # If requesting root, serve index.html
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

def serve_frontend(port=3002):
    """Serve the frontend on specified port"""
    # Change to frontend directory
    frontend_dir = "/home/amir/Documents/amir/CATO MAIOR/project/frontend"
    os.chdir(frontend_dir)
    
    # Create server
    handler = CustomHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 Frontend server starting on http://localhost:{port}")
            print(f"📁 Serving files from: {frontend_dir}")
            print(f"🔗 Open: http://localhost:{port}")
            print("Press Ctrl+C to stop the server")
            
            # Start server
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except OSError as e:
        print(f"❌ Error starting server: {e}")
        if "Address already in use" in str(e):
            print(f"Port {port} is already in use. Try a different port.")

if __name__ == "__main__":
    serve_frontend(3002)
