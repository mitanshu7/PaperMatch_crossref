import re
from ast import literal_eval
from bs4 import BeautifulSoup

text = """<jats:title>Summary</jats:title><jats:p><jats:list list-type="order"><jats:list-item><jats:p>Maternal provisioning of offspring in response to environmental conditions (“maternal environmental effects”) has been argued as ‘the missing link’ in plant life histories. Although empirical evidence suggests that maternal responses to abiotic conditions are common, there is little understanding of the prevalence of maternal provisioning in competitive environments.</jats:p></jats:list-item><jats:list-item><jats:p>We tested how competition in two soil moisture environments affects maternal provisioning of offspring seed mass. Specifically, we varied conspecific frequency from 90% (intraspecific competition) to 10% (interspecific competition) for 15 pairs of annual plant species that occur in California.</jats:p></jats:list-item><jats:list-item><jats:p>We found that conspecific frequency affected maternal provisioning (seed mass) in 48% of species, and that these responses included both increased (20%) and decreased (24%) seed mass. In contrast, 68% of species responded to competition through changes in per capita fecundity (seed number), which generally decreased as conspecific frequency increased. The direction and magnitude of frequency-dependent seed mass depended on the identity of the competitor, even among species in which fecundity was not affected by competitor identity.</jats:p></jats:list-item><jats:list-item><jats:p><jats:italic>Synthesis</jats:italic>. Our research demonstrates how species responses to different competitive environments manifest through maternal provisioning, and that these responses alter previous estimates of environmental maternal effects and reproductive output; future study is needed to understand their combined effects on population and community dynamics.</jats:p></jats:list-item></jats:list></jats:p>"""

# regex = re.compile(r"(?:<jats:p>)[\w\W]+(?:</jats:p>)")

# print(regex.findall(text))

soup = BeautifulSoup(text, 'html.parser')
# print(soup.find('jats:p').get_text())

print(soup.find('wtf').get_text())

# from datetime import datetime

# a = datetime.strptime('2017-10-17T01:10:23Z', 
# print(int(a.strftime('%Y')))
# print(a)
