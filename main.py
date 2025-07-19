import re
from ast import literal_eval
from bs4 import BeautifulSoup

text = """<JATS1:p>This accessible and engaging text covering sketch, sitcom and comedy drama, alongside improvisation and stand-up, brings together a panoply of tools and techniques for creating short and long-form comedy narratives for live performance, TV and online.</JATS1:p>
          <JATS1:p>Referencing a broad range of comedy from both sides of the Atlantic, spanning several decades and including material on contemporary internet sketches, it offers all kinds of useful advice on creating comic narratives for stage and screen: using life experience as raw material; constructing comedy worlds; creating comic characters, their relationships and interactions; structuring sketches, scenes and routines; and developing and plotting stories.</JATS1:p>
          <JATS1:p>The book’s interviewees, from the UK and the USA, feature stand-ups, sketch comics, improvisers and TV comedy producers, and include Steve Kaplan, Hollywood comedy guru and author of The Hidden Tools of Comedy, Will Hines teacher and improviser from the Upright Citizens Brigade Theatre and Lucy Lumsden TV producer and former Controller of Comedy Commissioning for BBC.</JATS1:p>
          <JATS1:p>Written by “the ideal person to nurture new talent” (The Guardian), Creating Comedy Narratives for Stage &amp; Screen includes material you won’t find anywhere else and is a stimulating resource for comedy students and their teachers, with a range and a depth that will be appreciated by even the most eclectic and multi-hyphenated writers and performers.</JATS1:p>"""

# regex = re.compile(r"(?:<jats:p>)[\w\W]+(?:</jats:p>)")

# print(regex.findall(text))

soup = BeautifulSoup(text, 'html.parser')
# print(soup.find('jats:p').get_text())

print(soup.find('wtf').get_text())

# from datetime import datetime

# a = datetime.strptime('2017-10-17T01:10:23Z', 
# print(int(a.strftime('%Y')))
# print(a)
