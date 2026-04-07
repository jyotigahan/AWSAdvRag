"""Simple HTTP server that serves the dashboard and metrics API."""
import os
import json
import http.server
import socketserver

from metrics_store import get_all

PORT = 8050
DASHBOARD_DIR = os.path.dirname(__file__)


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/metrics":
            data = get_all()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard running at http://localhost:{PORT}/dashboard.html")
        httpd.serve_forever()
