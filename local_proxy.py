"""
Local reverse proxy: localhost:11436 → JupyterHub Ollama proxy
Adds Bearer token transparently so litellm/CrewAI need no auth config.
Run in a separate terminal: python local_proxy.py
"""
import os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests
from dotenv import load_dotenv

load_dotenv()

TARGET = os.getenv("JUPYTERHUB_PROXY", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
TOKEN  = os.getenv("JUPYTERHUB_TOKEN", "")
PORT   = int(os.getenv("LOCAL_PROXY_PORT", "11436"))


class ProxyHandler(BaseHTTPRequestHandler):
    def _proxy(self):
        url = TARGET + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length")}
        if TOKEN:
            headers["Authorization"] = f"Bearer {TOKEN}"

        body = None
        n = int(self.headers.get("Content-Length", 0))
        if n > 0:
            body = self.rfile.read(n)

        try:
            # Retry up to 5 times on 599 (JupyterHub nginx upstream timeout)
            for attempt in range(5):
                resp = requests.request(
                    method=self.command,
                    url=url,
                    headers=headers,
                    data=body,
                    stream=True,
                    timeout=600,
                )
                if resp.status_code != 599:
                    break
                print(f"  [retry {attempt+1}/5] 599 on {self.path}")
                time.sleep(3)
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-encoding"):
                    self.send_header(k, v)
            self.end_headers()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (ConnectionAbortedError, BrokenPipeError):
                        break  # client closed connection, stop sending
        except (ConnectionAbortedError, BrokenPipeError):
            pass  # client gave up, ignore
        except Exception as exc:
            try:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(str(exc).encode())
            except Exception:
                pass

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _proxy

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path} → {args[1] if len(args)>1 else ''}")


if __name__ == "__main__":
    print(f"[Proxy] localhost:{PORT} → {TARGET}")
    print(f"[Proxy] Token: {'SET' if TOKEN else 'NOT SET'}")
    ThreadingHTTPServer(("127.0.0.1", PORT), ProxyHandler).serve_forever()
