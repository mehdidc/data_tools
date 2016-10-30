from bs4 import BeautifulSoup
import urllib
import wget 
import subprocess
import random
import time

def process_page(n=1):
    r = urllib.urlopen('http://www.flaticon.com/packs/{}'.format(n)).read()
    soup = BeautifulSoup(r, "html.parser")
    for el in soup.find_all("a", class_="cattitle"):
        process_pack(el["href"])
        time.sleep(random.uniform(0, 1))

def process_pack(link):
    r = urllib.urlopen(link).read()
    soup = BeautifulSoup(r, "html.parser")
    for el in soup.find_all("a", class_="fi-downloadfile3 downloadButton"):
        href = el["data-href"]
        name = href[href.index("pack=") + len("pack="):]
        print(name)
        process = subprocess.call(["wget", href, "--output-document={}.zip".format(name)])

if __name__ == "__main__":
    for i in range(40):
        process_page(n=i)
