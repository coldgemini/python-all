import urllib
import socks
from sockshandler import SocksiPyHandler
from urllib.request import urlopen

URL = 'https://www.github.com'
PROXY_ADDRESS = "127.0.0.1:1080"
# proxy = urllib.request.ProxyHandler({"http": PROXY_ADDRESS})
# opener = urllib.request.build_opener(proxy)
# proxy = urllib.request.ProxyHandler({"http": PROXY_ADDRESS})
proxy = SocksiPyHandler(socks.SOCKS5, "127.0.0.1", 1080)
opener = urllib.request.build_opener(proxy)
urllib.request.install_opener(opener)

# response = urlopen('http://www.packtpub.com')
# response = urllib.request.urlopen('http://www.packtpub.com')
# response = urlopen('http://www.baidu.com')
# response = urlopen('http://www.google.com')
response = urlopen(URL)
print("Proxy server	returns	response headers: %s " % response.headers)
print(response.getheader('Content-Type'))
# print(response)
# print(response.status)

# print(response.readline())

# http_response = response
# if http_response.code == 200:
#     print(http_response.headers)
#     for key, value in http_response.getheaders():
#         print(key, value)
#
# from urllib.request import Request
#
# from urllib.request import urlopen
#
# req = Request('http://www.python.org')
# urlopen(req)
# print(req.get_header('User-agent'))
