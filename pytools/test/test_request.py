import socket
import socks
import requests

# http_proxy = "http://<ip_address>:<port>"
ip = "127.0.0.1"
port = "1080"
socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, ip, port)
socket.socket = socks.socksocket
socks5_proxy = "socks5://127.0.0.1:1080"
# proxy_dictionary = {"http": http_proxy}
proxy_dictionary = {"http": socks5_proxy, "https": socks5_proxy}
text = requests.get("http://www.google.com", proxies=proxy_dictionary).text
print(text)
